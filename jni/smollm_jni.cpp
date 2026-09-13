#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <atomic>

#include "llama.h"

#define LOG_TAG "smollm_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct ChatTurn {
    std::string role;
    std::string content;
};

struct SmolLMSession {
    llama_model   *model   = nullptr;
    llama_context *ctx     = nullptr;
    llama_sampler *sampler = nullptr;
    std::atomic<bool> stopRequested{false};
    std::vector<ChatTurn> history;
    bool systemPromptSet = false; // ensures we only inject the system turn once per session
};

static bool g_backend_initialized = false;

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_smollmtest_SmolLM_nativeLoadModel(
        JNIEnv *env, jobject /* this */, jstring modelPath, jint nThreads, jint nCtx,
        jfloat temperature, jint topK, jfloat topP, jfloat minP, jint seed) {

    if (!g_backend_initialized) {
        llama_backend_init();
        g_backend_initialized = true;
    }

    const char *path = env->GetStringUTFChars(modelPath, nullptr);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    llama_model *model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!model) {
        LOGE("Failed to load model");
        return 0;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = nCtx > 0 ? nCtx : 2048;
    ctx_params.n_threads = nThreads > 0 ? nThreads : 4;
    ctx_params.n_threads_batch = ctx_params.n_threads;

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOGE("Failed to create context");
        llama_model_free(model);
        return 0;
    }

    int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // --- Sampling parameter recommendations (validate/clamp these on the Java side too) ---
    // topK:  1-100. RECOMMENDED DEFAULT: 40.
    //        Lower (10-20) = more focused/deterministic, higher (80-100) = more variety.
    //        0 disables top-k filtering entirely (not recommended for a 360M model — too wild).
    // topP:  0.0-1.0. RECOMMENDED DEFAULT: 0.9.
    //        Lower (0.5-0.7) = terser/safer, higher (0.95+) = more creative, more risk of nonsense.
    // minP:  0.0-1.0. RECOMMENDED DEFAULT: 0.05, used as an ALTERNATIVE to top-p, not alongside it.
    //        Set minP > 0 and topP = 1.0 to use min-p mode instead. Min-p tends to behave better
    //        than top-p on small models because it doesn't over-prune when the model is uncertain.
    // seed:  any int32, or -1 for random (maps to LLAMA_DEFAULT_SEED). RECOMMENDED DEFAULT: -1.
    //        Fix a seed only when you want reproducible output for debugging/comparison.
    int32_t effectiveTopK = topK > 0 ? topK : 40;
    float   effectiveTopP = (topP > 0.0f && topP <= 1.0f) ? topP : 0.9f;
    float   effectiveMinP = (minP > 0.0f && minP <= 1.0f) ? minP : 0.0f; // 0 = disabled
    uint32_t effectiveSeed = seed >= 0 ? (uint32_t) seed : LLAMA_DEFAULT_SEED;

    // Sampler chain order matters: penalties must run BEFORE top-k/top-p/min-p/temp,
    // otherwise they'd penalize an already-truncated candidate set.
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
            n_vocab,
            64,     // penalty_last_n: how many recent tokens count toward the penalty
            1.1f,   // penalty_repeat: >1.0 discourages repeats, keep modest (never exceed ~1.2)
            0.0f,   // penalty_freq: disabled
            0.0f)); // penalty_present: disabled
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(effectiveTopK));

    // If minP is set (> 0), prefer it over top-p by leaving top-p effectively passthrough (1.0).
    if (effectiveMinP > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_min_p(effectiveMinP, 1));
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(effectiveTopP, 1));
    }

    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature > 0 ? temperature : 0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(effectiveSeed));

    auto *session = new SmolLMSession();
    session->model = model;
    session->ctx = ctx;
    session->sampler = sampler;

    LOGI("Model loaded successfully, session=%p (topK=%d topP=%.2f minP=%.2f seed=%u)",
         session, effectiveTopK, effectiveTopP, effectiveMinP, effectiveSeed);
    return reinterpret_cast<jlong>(session);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_smollmtest_SmolLM_nativeGenerate(
        JNIEnv *env, jobject thiz, jlong sessionPtr, jstring prompt,
        jint maxTokens, jstring systemPrompt, jobject callback) {

    auto *session = reinterpret_cast<SmolLMSession *>(sessionPtr);
    if (!session || !session->ctx || !session->model) {
        LOGE("Invalid session in nativeGenerate");
        return;
    }

    session->stopRequested = false;

    // Inject the system prompt ONCE, as the very first history entry, before any user turn.
    // RECOMMENDED DEFAULT: empty/none. A short, specific instruction ("You are a concise
    // assistant. Answer in 2-3 sentences unless asked for more.") tends to help small models
    // stay on-task more than long/elaborate personas, which they can lose track of.
    if (!session->systemPromptSet && systemPrompt != nullptr) {
        const char *sys_cstr = env->GetStringUTFChars(systemPrompt, nullptr);
        std::string sys_str(sys_cstr);
        env->ReleaseStringUTFChars(systemPrompt, sys_cstr);
        if (!sys_str.empty()) {
            session->history.insert(session->history.begin(), {"system", sys_str});
        }
        session->systemPromptSet = true;
    }

    const char *prompt_cstr = env->GetStringUTFChars(prompt, nullptr);
    std::string user_prompt(prompt_cstr);
    env->ReleaseStringUTFChars(prompt, prompt_cstr);

    // Append this turn to persistent history BEFORE formatting, so the model sees full context.
    session->history.push_back({"user", user_prompt});

    // Build the llama_chat_message array from history. Pointers must stay valid
    // for the duration of this call, which they are since `history` strings are stable.
    std::vector<llama_chat_message> chat_msgs;
    chat_msgs.reserve(session->history.size());
    for (const auto &turn : session->history) {
        chat_msgs.push_back({turn.role.c_str(), turn.content.c_str()});
    }

    const char *tmpl = llama_model_chat_template(session->model, nullptr);

    std::vector<char> formatted(user_prompt.size() * 4 + 1024);
    int32_t formatted_len = llama_chat_apply_template(
            tmpl,
            chat_msgs.data(),
            chat_msgs.size(),
            true, // add_ass
            formatted.data(),
            (int32_t) formatted.size());

    std::string prompt_str;
    if (formatted_len < 0) {
        LOGE("llama_chat_apply_template failed (tmpl=%s), falling back to raw prompt",
             tmpl ? tmpl : "null");
        prompt_str = user_prompt;
    } else {
        if ((size_t) formatted_len > formatted.size()) {
            formatted.resize(formatted_len);
            llama_chat_apply_template(tmpl, chat_msgs.data(), chat_msgs.size(), true,
                                       formatted.data(), formatted_len);
        }
        prompt_str.assign(formatted.data(), formatted_len);
    }

    LOGI("Formatted prompt (%zu history turns): %s", session->history.size(), prompt_str.c_str());

    const llama_vocab *vocab = llama_model_get_vocab(session->model);

    int n_prompt_tokens = -llama_tokenize(vocab, prompt_str.c_str(), (int32_t) prompt_str.size(),
                                           nullptr, 0, true, true);
    if (n_prompt_tokens <= 0) {
        LOGE("Prompt tokenization returned invalid size: %d", n_prompt_tokens);
        session->history.pop_back(); // roll back the turn we couldn't process
        return;
    }

    std::vector<llama_token> tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt_str.c_str(), (int32_t) prompt_str.size(),
                        tokens.data(), (int32_t) tokens.size(), true, true) < 0) {
        LOGE("Tokenization failed");
        session->history.pop_back();
        return;
    }

    // Guard against exceeding context size as history grows over many turns.
    int n_ctx = llama_n_ctx(session->ctx);
    if ((int) tokens.size() >= n_ctx - 8) {
        LOGE("Prompt+history (%d tokens) too close to context limit (%d). Trimming oldest turns.",
             (int) tokens.size(), n_ctx);
        // Simple strategy: drop oldest turns (keep at least the current one) and retry once.
        // NOTE: this could now drop the system turn if history grows huge — acceptable tradeoff
        // for this simple strategy, flagged here in case it matters later.
        while (session->history.size() > 1 && (int) tokens.size() >= n_ctx - 8) {
            session->history.erase(session->history.begin());
            chat_msgs.clear();
            for (const auto &turn : session->history) {
                chat_msgs.push_back({turn.role.c_str(), turn.content.c_str()});
            }
            formatted_len = llama_chat_apply_template(tmpl, chat_msgs.data(), chat_msgs.size(),
                                                        true, formatted.data(), (int32_t) formatted.size());
            if (formatted_len < 0) break;
            prompt_str.assign(formatted.data(), formatted_len);
            n_prompt_tokens = -llama_tokenize(vocab, prompt_str.c_str(), (int32_t) prompt_str.size(),
                                               nullptr, 0, true, true);
            tokens.resize(n_prompt_tokens);
            llama_tokenize(vocab, prompt_str.c_str(), (int32_t) prompt_str.size(),
                            tokens.data(), (int32_t) tokens.size(), true, true);
        }
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());

    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onTokenMethod = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");

    char piece_buf[256];
    std::string assistantReply;

    for (int i = 0; i < maxTokens; i++) {
        if (session->stopRequested.load()) {
            LOGI("Stop requested, halting generation at step %d", i);
            break;
        }

        if (llama_decode(session->ctx, batch) != 0) {
            LOGE("llama_decode failed at step %d", i);
            break;
        }

        llama_token new_token = llama_sampler_sample(session->sampler, session->ctx, -1);
        llama_sampler_accept(session->sampler, new_token); // feeds the penalty tracker

        if (llama_vocab_is_eog(vocab, new_token)) {
            LOGI("EOG token reached, stopping generation");
            break;
        }

        int n = llama_token_to_piece(vocab, new_token, piece_buf, sizeof(piece_buf), 0, true);
        if (n < 0) {
            LOGE("token_to_piece failed");
            break;
        }
        std::string piece(piece_buf, n);
        assistantReply += piece;

        jstring jpiece = env->NewStringUTF(piece.c_str());
        env->CallVoidMethod(callback, onTokenMethod, jpiece);
        env->DeleteLocalRef(jpiece);

        batch = llama_batch_get_one(&new_token, 1);
    }

    // Persist the assistant's reply into history so the NEXT turn has full context.
    session->history.push_back({"assistant", assistantReply});
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_smollmtest_SmolLM_nativeClearHistory(
        JNIEnv *env, jobject /* this */, jlong sessionPtr) {

    auto *session = reinterpret_cast<SmolLMSession *>(sessionPtr);
    if (session) {
        session->history.clear();
        session->systemPromptSet = false; // allow a fresh system prompt on next generate
        LOGI("Conversation history cleared for session=%p", session);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_smollmtest_SmolLM_nativeStop(
        JNIEnv *env, jobject /* this */, jlong sessionPtr) {

    auto *session = reinterpret_cast<SmolLMSession *>(sessionPtr);
    if (session) {
        session->stopRequested = true;
        LOGI("Stop flag set for session=%p", session);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_smollmtest_SmolLM_nativeUnload(
        JNIEnv *env, jobject /* this */, jlong sessionPtr) {

    auto *session = reinterpret_cast<SmolLMSession *>(sessionPtr);
    if (!session) return;

    if (session->sampler) llama_sampler_free(session->sampler);
    if (session->ctx)     llama_free(session->ctx);
    if (session->model)   llama_model_free(session->model);

    delete session;
    LOGI("Session unloaded");
}
