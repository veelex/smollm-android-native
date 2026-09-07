#include <jni.h>
#include <android/log.h>

#define LOG_TAG "smollm_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_smollmtest_SmolLM_nativePing(JNIEnv *env, jobject /* this */) {
    LOGI("nativePing() called");
    return env->NewStringUTF("pong from native");
}
