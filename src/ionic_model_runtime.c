/*
 * Ionic ML model runtime
 *
 * Supported backends:
 *   .onnx              — ONNX Runtime (CoreML GPU on macOS)
 *   .gguf              — llama.cpp (Metal GPU on macOS)
 *   .pt / .pth         — stub (LibTorch not yet linked)
 *   .h5                — stub (Keras/HDF5 not yet linked)
 *   .mlmodel           — stub (CoreML direct load not yet linked)
 *
 * Public API:
 *   ionic_load_model(path)                                 -> opaque handle
 *   ionic_model_free(model)                                -> void
 *   ionic_model_forward(model, input)                      -> tensor ptr (ONNX single-input)
 *   ionic_piper_forward(model, phoneme_arr, ns, ls, nw)    -> float64 array
 *   ionic_write_wav(path, arr, n_samples, sample_rate)     -> void
 *   ionic_gguf_generate(model, prompt, max_tokens)         -> string (char*)
 *   ionic_gguf_set_temp(model, temp)                       -> void
 *   ionic_gguf_set_top_p(model, top_p)                     -> void
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#ifdef _WIN32
#  include <windows.h>
#else
#  include <unistd.h>
#endif

/* ── ONNX Runtime ─────────────────────────────────────────────────────────── */
#ifdef IONIC_HAVE_ORT
#  include "onnxruntime_c_api.h"
#  ifdef __APPLE__
#    include "coreml_provider_factory.h"
#  endif
#endif

/* ── llama.cpp (GGUF) ─────────────────────────────────────────────────────── */
#ifdef IONIC_HAVE_LLAMA
#  include "llama.h"
#endif

/* ── Ionic dynamic-array layout ──────────────────────────────────────────────
 *   struct IonicArray { i64 len; i64 cap; void *data; }
 *   data is an array of i64 slots (8 bytes each)
 * ─────────────────────────────────────────────────────────────────────────── */
typedef struct {
    int64_t len;
    int64_t cap;
    void   *data;
} IonicArray;

static IonicArray *ionic_array_alloc(int64_t cap) {
    IonicArray *a = (IonicArray *)malloc(sizeof(IonicArray));
    a->len  = 0;
    a->cap  = cap > 0 ? cap : 8;
    a->data = malloc((size_t)(a->cap * 8));
    return a;
}

static void ionic_array_push_i64(IonicArray *a, int64_t v) {
    if (a->len >= a->cap) {
        a->cap *= 2;
        a->data = realloc(a->data, (size_t)(a->cap * 8));
    }
    ((int64_t *)a->data)[a->len++] = v;
}

/* ── System runtime ───────────────────────────────────────────────────────── */

static int    ionic_argc = 0;
static char **ionic_argv = NULL;

char *ionic_fgets_stdin(char *buf, int n) {
    char *r = fgets(buf, n, stdin);
    if (!r) { buf[0] = '\0'; return buf; }
    /* Strip trailing newline */
    size_t len = strlen(buf);
    if (len > 0 && buf[len-1] == '\n') buf[len-1] = '\0';
    return buf;
}

void ionic_runtime_init(int argc, char **argv) {
    ionic_argc = argc;
    ionic_argv = argv;
}

const char *ionic_get_arg(int64_t n) {
    int idx = (int)(n + 1);
    if (idx < 1 || idx >= ionic_argc || !ionic_argv[idx]) return "";
    return ionic_argv[idx];
}

int64_t ionic_cpu_core_count(void) {
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si);
    return (int64_t)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int64_t)n : 1;
#endif
}

/* ── Model format detection ───────────────────────────────────────────────── */

typedef enum {
    MODEL_FMT_UNKNOWN = 0,
    MODEL_FMT_ONNX,
    MODEL_FMT_GGUF,
    MODEL_FMT_PT,
    MODEL_FMT_H5,
    MODEL_FMT_MLMODEL,
} ModelFormat;

static ModelFormat detect_format(const char *path) {
    /* Extension check first */
    const char *dot = strrchr(path, '.');
    if (dot) {
        if (strcmp(dot, ".onnx")    == 0) return MODEL_FMT_ONNX;
        if (strcmp(dot, ".gguf")    == 0) return MODEL_FMT_GGUF;
        if (strcmp(dot, ".pt")      == 0) return MODEL_FMT_PT;
        if (strcmp(dot, ".pth")     == 0) return MODEL_FMT_PT;
        if (strcmp(dot, ".h5")      == 0) return MODEL_FMT_H5;
        if (strcmp(dot, ".mlmodel") == 0) return MODEL_FMT_MLMODEL;
    }
    /* Magic-byte fallback for extension-less files (e.g. Ollama blobs) */
    FILE *f = fopen(path, "rb");
    if (!f) return MODEL_FMT_UNKNOWN;
    unsigned char magic[8] = {0};
    fread(magic, 1, sizeof(magic), f);
    fclose(f);
    /* GGUF: "GGUF" */
    if (magic[0]=='G' && magic[1]=='G' && magic[2]=='U' && magic[3]=='F') return MODEL_FMT_GGUF;
    /* ONNX protobuf: starts with 0x08 field-tag (common for ModelProto) — heuristic only */
    /* ORT also has its own binary format; skip for now */
    return MODEL_FMT_UNKNOWN;
}

/* ════════════════════════════════════════════════════════════════════════════
 * ONNX Runtime backend
 * ════════════════════════════════════════════════════════════════════════════ */

#ifdef IONIC_HAVE_ORT

static const OrtApi *g_ort = NULL;

static void ort_check(OrtStatus *status, const char *ctx) {
    if (!status) return;
    const char *msg = g_ort->GetErrorMessage(status);
    fprintf(stderr, "[ionic/ort] %s: %s\n", ctx, msg);
    g_ort->ReleaseStatus(status);
    exit(1);
}

static void ensure_ort(void) {
    if (g_ort) return;
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) { fprintf(stderr, "[ionic/ort] API init failed\n"); exit(1); }
}

typedef struct {
    ModelFormat  fmt;       /* MODEL_FMT_ONNX */
    OrtEnv      *env;
    OrtSession  *session;
} OnnxHandle;

static void *ort_load(const char *path) {
    ensure_ort();
    OnnxHandle *h = calloc(1, sizeof(OnnxHandle));
    h->fmt = MODEL_FMT_ONNX;
    ort_check(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ionic", &h->env), "CreateEnv");
    OrtSessionOptions *opts = NULL;
    ort_check(g_ort->CreateSessionOptions(&opts), "CreateSessionOptions");
#ifdef __APPLE__
    OrtStatus *cml_st = OrtSessionOptionsAppendExecutionProvider_CoreML(opts, 0);
    if (cml_st) { g_ort->ReleaseStatus(cml_st); }
    else        { fprintf(stderr, "[ionic/ort] CoreML EP enabled\n"); }
#endif
    ort_check(g_ort->CreateSession(h->env, path, opts, &h->session), "CreateSession");
    g_ort->ReleaseSessionOptions(opts);
    fprintf(stderr, "[ionic/ort] loaded: %s\n", path);
    return h;
}

static void ort_free(void *model) {
    ensure_ort();
    OnnxHandle *h = model;
    if (h->session) g_ort->ReleaseSession(h->session);
    if (h->env)     g_ort->ReleaseEnv(h->env);
    free(h);
}

#endif /* IONIC_HAVE_ORT */

/* ════════════════════════════════════════════════════════════════════════════
 * llama.cpp / GGUF backend
 * ════════════════════════════════════════════════════════════════════════════ */

#ifdef IONIC_HAVE_LLAMA

static int g_llama_backend_init = 0;

static void ensure_llama_backend(void) {
    if (g_llama_backend_init) return;
    llama_backend_init();
    g_llama_backend_init = 1;
}

typedef struct {
    ModelFormat          fmt;        /* MODEL_FMT_GGUF */
    struct llama_model  *model;
    struct llama_context*ctx;
    float                temperature; /* sampling temperature */
    float                top_p;       /* nucleus sampling p   */
} GgufHandle;

static void *gguf_load(const char *path) {
    ensure_llama_backend();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999; /* offload all layers to Metal */

    struct llama_model *model = llama_model_load_from_file(path, mparams);
    if (!model) {
        fprintf(stderr, "[ionic/llama] failed to load: %s\n", path);
        return NULL;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 4096;
    cparams.n_batch = 512;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[ionic/llama] failed to create context\n");
        llama_model_free(model);
        return NULL;
    }

    GgufHandle *h = calloc(1, sizeof(GgufHandle));
    h->fmt         = MODEL_FMT_GGUF;
    h->model       = model;
    h->ctx         = ctx;
    h->temperature = 0.8f;
    h->top_p       = 0.95f;

    fprintf(stderr, "[ionic/llama] loaded: %s\n", path);
    return h;
}

static void ionic_gguf_handle_free(void *model) {
    GgufHandle *h = model;
    /* Order matters: free context before model, then backend to drain Metal residency sets */
    if (h->ctx)   { llama_free(h->ctx);         h->ctx   = NULL; }
    if (h->model) { llama_model_free(h->model);  h->model = NULL; }
    llama_backend_free();
    g_llama_backend_init = 0;
    free(h);
}

/* Generate text from a prompt; returns a malloc'd string owned by the runtime */
static const char *gguf_generate(void *model, const char *prompt, int64_t max_tokens) {
    GgufHandle *h = model;
    const struct llama_vocab *vocab = llama_model_get_vocab(h->model);

    /* Tokenise prompt */
    int n_prompt_tokens = -llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
                                           NULL, 0, 1, 1);
    llama_token *prompt_tokens = malloc((size_t)n_prompt_tokens * sizeof(llama_token));
    llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
                   prompt_tokens, n_prompt_tokens, 1, 1);

    /* KV-cache reset */
    llama_memory_clear(llama_get_memory(h->ctx), 1);

    /* Build sampler chain */
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler *smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(h->top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(h->temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    /* Decode prompt in one batch */
    struct llama_batch batch = llama_batch_get_one(prompt_tokens, n_prompt_tokens);
    if (llama_decode(h->ctx, batch) != 0) {
        fprintf(stderr, "[ionic/llama] prompt decode failed\n");
        llama_sampler_free(smpl);
        free(prompt_tokens);
        return "";
    }

    /* Accumulate output into a growable buffer */
    size_t buf_cap = 4096;
    size_t buf_len = 0;
    char  *buf     = malloc(buf_cap);
    buf[0] = '\0';

    char piece[256];
    int64_t generated = 0;

    while (generated < max_tokens) {
        llama_token tok = llama_sampler_sample(smpl, h->ctx, -1);

        if (llama_vocab_is_eog(vocab, tok)) break;

        int n = llama_token_to_piece(vocab, tok, piece, sizeof(piece) - 1, 0, 1);
        if (n < 0) n = 0;
        piece[n] = '\0';

        /* Grow buffer if needed */
        while (buf_len + (size_t)n + 1 > buf_cap) {
            buf_cap *= 2;
            buf = realloc(buf, buf_cap);
        }
        memcpy(buf + buf_len, piece, (size_t)n);
        buf_len += (size_t)n;
        buf[buf_len] = '\0';

        /* Decode the new token */
        llama_token new_tok = tok;
        struct llama_batch next = llama_batch_get_one(&new_tok, 1);
        if (llama_decode(h->ctx, next) != 0) break;

        generated++;
    }

    llama_sampler_free(smpl);
    free(prompt_tokens);

    fprintf(stderr, "[ionic/llama] generated %lld tokens\n", (long long)generated);
    return buf; /* caller owns — but Ionic treats strings as const; leak is acceptable for now */
}

#endif /* IONIC_HAVE_LLAMA */

/* ════════════════════════════════════════════════════════════════════════════
 * Public API — dispatch on ModelFormat stored in handle
 * ════════════════════════════════════════════════════════════════════════════ */

/* The first field of every handle struct is ModelFormat fmt — safe to cast */
static ModelFormat handle_fmt(void *model) {
    if (!model) return MODEL_FMT_UNKNOWN;
    return *(ModelFormat *)model;
}

void *ionic_load_model(const char *path) {
    ModelFormat fmt = detect_format(path);
    switch (fmt) {
#ifdef IONIC_HAVE_ORT
        case MODEL_FMT_ONNX:
            return ort_load(path);
#endif
#ifdef IONIC_HAVE_LLAMA
        case MODEL_FMT_GGUF:
            return gguf_load(path);
#endif
        case MODEL_FMT_PT:
            fprintf(stderr, "[ionic] load_model: PyTorch (.pt/.pth) not yet supported\n");
            return NULL;
        case MODEL_FMT_H5:
            fprintf(stderr, "[ionic] load_model: Keras/HDF5 (.h5) not yet supported\n");
            return NULL;
        case MODEL_FMT_MLMODEL:
            fprintf(stderr, "[ionic] load_model: CoreML (.mlmodel) not yet supported\n");
            return NULL;
        default:
            fprintf(stderr, "[ionic] load_model: unknown format '%s'\n", path);
            return NULL;
    }
}

void ionic_model_free(void *model) {
    if (!model) return;
    switch (handle_fmt(model)) {
#ifdef IONIC_HAVE_ORT
        case MODEL_FMT_ONNX: ort_free(model); break;
#endif
#ifdef IONIC_HAVE_LLAMA
        case MODEL_FMT_GGUF: ionic_gguf_handle_free(model); break;
#endif
        default: free(model); break;
    }
}

void *ionic_model_forward(void *model, void *input) {
    if (!model) return input;
    fprintf(stderr, "[ionic] ionic_model_forward: use ionic_piper_forward for ONNX, ionic_gguf_generate for GGUF\n");
    return input;
}

/* ── GGUF text generation ─────────────────────────────────────────────────── */

const char *ionic_gguf_generate(void *model, const char *prompt, int64_t max_tokens) {
#ifdef IONIC_HAVE_LLAMA
    if (!model || handle_fmt(model) != MODEL_FMT_GGUF) {
        fprintf(stderr, "[ionic] gguf_generate: not a GGUF model\n");
        return "";
    }
    return gguf_generate(model, prompt, max_tokens);
#else
    (void)model; (void)prompt; (void)max_tokens;
    fprintf(stderr, "[ionic] gguf_generate: llama.cpp not compiled in\n");
    return "";
#endif
}

void ionic_gguf_set_temp(void *model, double temp) {
#ifdef IONIC_HAVE_LLAMA
    if (model && handle_fmt(model) == MODEL_FMT_GGUF)
        ((GgufHandle *)model)->temperature = (float)temp;
#else
    (void)model; (void)temp;
#endif
}

void ionic_gguf_set_top_p(void *model, double top_p) {
#ifdef IONIC_HAVE_LLAMA
    if (model && handle_fmt(model) == MODEL_FMT_GGUF)
        ((GgufHandle *)model)->top_p = (float)top_p;
#else
    (void)model; (void)top_p;
#endif
}

/* ── Piper TTS forward (ONNX) ─────────────────────────────────────────────── */

void *ionic_piper_forward(void *model,
                          void *phoneme_arr,
                          double noise_scale,
                          double length_scale,
                          double noise_w) {
#ifdef IONIC_HAVE_ORT
    ensure_ort();
    if (!model || handle_fmt(model) != MODEL_FMT_ONNX) {
        fprintf(stderr, "[ionic] piper_forward: not an ONNX model\n");
        return ionic_array_alloc(1);
    }

    OnnxHandle  *h   = model;
    IonicArray  *arr = phoneme_arr;
    int64_t      T   = arr->len;
    int64_t     *ids = (int64_t *)arr->data;

    OrtMemoryInfo *mem_info = NULL;
    ort_check(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info),
              "CreateCpuMemoryInfo");

    int64_t shape_input[2] = {1, T};
    OrtValue *ov_input = NULL;
    ort_check(g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, ids, (size_t)(T * sizeof(int64_t)),
        shape_input, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &ov_input), "input tensor");

    int64_t lengths_data[1] = {T};
    int64_t shape_len[1]    = {1};
    OrtValue *ov_lengths = NULL;
    ort_check(g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, lengths_data, sizeof(int64_t),
        shape_len, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &ov_lengths), "lengths tensor");

    float scales_data[3] = {(float)noise_scale, (float)length_scale, (float)noise_w};
    int64_t shape_scales[1] = {3};
    OrtValue *ov_scales = NULL;
    ort_check(g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, scales_data, 3 * sizeof(float),
        shape_scales, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ov_scales), "scales tensor");

    const char *input_names[]  = {"input", "input_lengths", "scales"};
    const char *output_names[] = {"output"};
    OrtValue   *inputs[3]      = {ov_input, ov_lengths, ov_scales};
    OrtValue   *ov_output      = NULL;
    ort_check(g_ort->Run(h->session, NULL,
        input_names, (const OrtValue *const *)inputs, 3,
        output_names, 1, &ov_output), "Run");

    float *out_data = NULL;
    ort_check(g_ort->GetTensorMutableData(ov_output, (void **)&out_data), "GetTensorMutableData");

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort_check(g_ort->GetTensorTypeAndShape(ov_output, &shape_info), "GetTensorTypeAndShape");
    size_t elem_count = 0;
    ort_check(g_ort->GetTensorShapeElementCount(shape_info, &elem_count), "GetElementCount");
    g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    IonicArray *result = ionic_array_alloc((int64_t)elem_count);
    for (size_t i = 0; i < elem_count; i++) {
        double d = (double)out_data[i];
        int64_t bits;
        memcpy(&bits, &d, 8);
        ionic_array_push_i64(result, bits);
    }

    fprintf(stderr, "[ionic/ort] piper: %zu samples\n", elem_count);

    g_ort->ReleaseValue(ov_output);
    g_ort->ReleaseValue(ov_scales);
    g_ort->ReleaseValue(ov_lengths);
    g_ort->ReleaseValue(ov_input);
    g_ort->ReleaseMemoryInfo(mem_info);
    return result;
#else
    (void)model; (void)phoneme_arr;
    (void)noise_scale; (void)length_scale; (void)noise_w;
    fprintf(stderr, "[ionic] piper_forward: ONNX Runtime not compiled in\n");
    return ionic_array_alloc(1);
#endif
}

/* ── WAV writer ───────────────────────────────────────────────────────────── */

static void write_le16(FILE *f, uint16_t v) { fputc(v & 0xff, f); fputc(v >> 8, f); }
static void write_le32(FILE *f, uint32_t v) {
    fputc(v & 0xff, f); fputc((v >> 8) & 0xff, f);
    fputc((v >> 16) & 0xff, f); fputc((v >> 24) & 0xff, f);
}

void ionic_write_wav(const char *path, void *samples_arr,
                     int64_t num_samples, int64_t sample_rate) {
    IonicArray *arr = samples_arr;
    int64_t     n   = (num_samples > 0 && num_samples <= arr->len) ? num_samples : arr->len;

    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[ionic] write_wav: cannot open '%s'\n", path); return; }

    uint32_t data_bytes = (uint32_t)(n * 2);
    fwrite("RIFF", 1, 4, f); write_le32(f, 36 + data_bytes);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); write_le32(f, 16);
    write_le16(f, 1);                        /* PCM */
    write_le16(f, 1);                        /* mono */
    write_le32(f, (uint32_t)sample_rate);
    write_le32(f, (uint32_t)(sample_rate * 2)); /* byte rate */
    write_le16(f, 2);                        /* block align */
    write_le16(f, 16);                       /* bits per sample */
    fwrite("data", 1, 4, f); write_le32(f, data_bytes);

    int64_t *slots = arr->data;
    for (int64_t i = 0; i < n; i++) {
        double d; memcpy(&d, &slots[i], 8);
        if (d >  1.0) d =  1.0;
        if (d < -1.0) d = -1.0;
        int16_t s = (int16_t)(d * 32767.0);
        write_le16(f, (uint16_t)s);
    }
    fclose(f);
    fprintf(stderr, "[ionic] WAV: %s (%lld samples @ %lld Hz)\n",
            path, (long long)n, (long long)sample_rate);
}
