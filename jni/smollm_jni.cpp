#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <atomic>

#include "llama.h"

#define LOG_TAG "smollm_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct SmolLMSession {
    llama_model   *model   = nullptr;
    llama_context *ctx     = nullptr;
    llama_sampler *sampler = nullptr;
    std::atomic<bool> stopRequested{false};
};

static bool g_backend_initialized = false;

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_smollmtest_SmolLM_nativeLoadModel(
        JNIEnv *env, jobject /* this */, jstring modelPath, jint nThreads, jint nCtx,
        jfloat temperature) {

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

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature > 0 ? temperature : 0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    auto *session = new SmolLMSession();
    session->model = model;
    session->ctx = ctx;
    session->sampler = sampler;

    LOGI("Model loaded successfully, session=%p", session);
    return reinterpret_cast<jlong>(session);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_smollmtest_SmolLM_nativeGenerate(
        JNIEnv *env, jobject thiz, jlong sessionPtr, jstring prompt,
        jint maxTokens, jobject callback) {

    auto *session = reinterpret_cast<SmolLMSession *>(sessionPtr);
    if (!session || !session->ctx || !session->model) {
        LOGE("Invalid session in nativeGenerate");
        return;
    }

    session->stopRequested = false;

    const char *prompt_cstr = env->GetStringUTFChars(prompt, nullptr);
    std::string user_prompt(prompt_cstr);
    env->ReleaseStringUTFChars(prompt, prompt_cstr);

    // --- Apply the model's chat template so it recognizes this as an instruct turn ---
    llama_chat_message chat_msg[1];
    chat_msg[0].role = "user";
    chat_msg[0].content = user_prompt.c_str();

    std::vector<char> formatted(user_prompt.size() * 4 + 256);
    int32_t formatted_len = llama_chat_apply_template(
            session->model,
            nullptr,   // nullptr = use the template embedded in the GGUF metadata
            chat_msg,
            1,
            true,      // add_ass: append the assistant-turn prefix, so the model knows to reply next
            formatted.data(),
            (int32_t) formatted.size());

    std::string prompt_str;
    if (formatted_len < 0) {
        LOGE("llama_chat_apply_template failed, falling back to raw prompt");
        prompt_str = user_prompt;
    } else {
        if ((size_t) formatted_len > formatted.size()) {
            formatted.resize(formatted_len);
            llama_chat_apply_template(session->model, nullptr, chat_msg, 1, true,
                                       formatted.data(), formatted_len);
        }
        prompt_str.assign(formatted.data(), formatted_len);
    }

    LOGI("Formatted prompt: %s", prompt_str.c_str());

    const llama_vocab *vocab = llama_model_get_vocab(session->model);

    int n_prompt_tokens = -llama_tokenize(vocab, prompt_str.c_str(), (int32_t) prompt_str.size(),
                                           nullptr, 0, true, true);
    if (n_prompt_tokens <= 0) {
        LOGE("Prompt tokenization returned invalid size: %d", n_prompt_tokens);
        return;
    }

    std::vector<llama_token> tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt_str.c_str(), (int32_t) prompt_str.size(),
                        tokens.data(), (int32_t) tokens.size(), true, true) < 0) {
        LOGE("Tokenization failed");
        return;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());

    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onTokenMethod = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");

    char piece_buf[256];

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

        jstring jpiece = env->NewStringUTF(piece.c_str());
        env->CallVoidMethod(callback, onTokenMethod, jpiece);
        env->DeleteLocalRef(jpiece);

        batch = llama_batch_get_one(&new_token, 1);
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
