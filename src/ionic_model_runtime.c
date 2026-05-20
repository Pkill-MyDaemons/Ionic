/*
 * Ionic ML model runtime — Phase 1 stubs
 *
 * ionic_load_model(path)          -> opaque handle (malloc'd ModelHandle)
 * ionic_model_forward(model, in)  -> tensor ptr (stub: returns input unchanged)
 * ionic_model_free(model)         -> void
 *
 * Full implementations require:
 *   .pt / .pth  — LibTorch C++ API  (link -ltorch)
 *   .onnx       — ONNX Runtime C API (link -lonnxruntime)
 *   .h5         — Keras/HDF5 via Python bridge or cHDF5 (link -lhdf5)
 *   .mlmodel    — Apple CoreML (macOS only, Objective-C / Swift bridge)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef _WIN32
#  include <windows.h>
#else
#  include <unistd.h>
#endif

/* ── System runtime ─────────────────────────────────────────────────────────── */

static int    ionic_argc = 0;
static char **ionic_argv = NULL;

void ionic_runtime_init(int argc, char **argv) {
    ionic_argc = argc;
    ionic_argv = argv;
}

/* get_arg(n): returns argv[n+1] (skips program name), or "" if out of range */
const char *ionic_get_arg(int64_t n) {
    int idx = (int)(n + 1);
    if (idx < 1 || idx >= ionic_argc || !ionic_argv[idx]) return "";
    return ionic_argv[idx];
}

/* cpu_core_count(): returns number of logical CPUs available */
int64_t ionic_cpu_core_count(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int64_t)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int64_t)n : 1;
#endif
}

typedef enum {
    MODEL_FMT_UNKNOWN = 0,
    MODEL_FMT_PT,       /* PyTorch .pt / .pth */
    MODEL_FMT_ONNX,     /* ONNX Runtime .onnx */
    MODEL_FMT_H5,       /* Keras HDF5 .h5     */
    MODEL_FMT_MLMODEL,  /* Apple CoreML .mlmodel */
} ModelFormat;

typedef struct {
    ModelFormat fmt;
    void       *handle;  /* format-specific handle; NULL in Phase 1 stubs */
    char        path[1024];
} ModelHandle;

static ModelFormat detect_format(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot) return MODEL_FMT_UNKNOWN;
    if (strcmp(dot, ".pt")  == 0 || strcmp(dot, ".pth") == 0) return MODEL_FMT_PT;
    if (strcmp(dot, ".onnx") == 0)    return MODEL_FMT_ONNX;
    if (strcmp(dot, ".h5")   == 0)    return MODEL_FMT_H5;
    if (strcmp(dot, ".mlmodel") == 0) return MODEL_FMT_MLMODEL;
    return MODEL_FMT_UNKNOWN;
}

/*
 * Load a model from disk.
 * Returns an opaque ModelHandle*, or NULL on failure.
 */
void *ionic_load_model(const char *path) {
    ModelFormat fmt = detect_format(path);
    if (fmt == MODEL_FMT_UNKNOWN) {
        fprintf(stderr, "[ionic] load_model: unknown format for '%s'\n", path);
        return NULL;
    }

    ModelHandle *m = (ModelHandle *)malloc(sizeof(ModelHandle));
    if (!m) return NULL;
    m->fmt    = fmt;
    m->handle = NULL;
    strncpy(m->path, path, sizeof(m->path) - 1);
    m->path[sizeof(m->path) - 1] = '\0';

    switch (fmt) {
        case MODEL_FMT_PT:
            /* TODO: link LibTorch and call torch::jit::load(path) */
            fprintf(stderr, "[ionic] load_model: PyTorch stub — '%s' not actually loaded\n", path);
            break;
        case MODEL_FMT_ONNX:
            /* TODO: link ONNX Runtime C API and create OrtSession */
            fprintf(stderr, "[ionic] load_model: ONNX stub — '%s' not actually loaded\n", path);
            break;
        case MODEL_FMT_H5:
            /* TODO: HDF5 + Keras layer reconstruction */
            fprintf(stderr, "[ionic] load_model: Keras/H5 stub — '%s' not actually loaded\n", path);
            break;
        case MODEL_FMT_MLMODEL:
            /* TODO: Apple CoreML MLModel compile + load (macOS only) */
            fprintf(stderr, "[ionic] load_model: CoreML stub — '%s' not actually loaded\n", path);
            break;
        default:
            break;
    }

    return (void *)m;
}

/*
 * Run a forward pass.
 * model — ModelHandle* from ionic_load_model
 * input — tensor ptr (ionic dynamic-array header)
 * Returns a tensor ptr (Phase 1: returns input unchanged as a stub).
 */
void *ionic_model_forward(void *model, void *input) {
    if (!model) {
        fprintf(stderr, "[ionic] model_forward: null model handle\n");
        return input;
    }
    ModelHandle *m = (ModelHandle *)model;
    switch (m->fmt) {
        case MODEL_FMT_PT:
            /* TODO: call torch::jit::Module::forward() */
            break;
        case MODEL_FMT_ONNX:
            /* TODO: call OrtSession::Run() */
            break;
        case MODEL_FMT_H5:
            /* TODO: run Keras model */
            break;
        case MODEL_FMT_MLMODEL:
            /* TODO: MLModel prediction */
            break;
        default:
            break;
    }
    /* Phase 1 stub: return input tensor unchanged */
    return input;
}

/*
 * Free a model handle and any associated resources.
 */
void ionic_model_free(void *model) {
    if (!model) return;
    ModelHandle *m = (ModelHandle *)model;
    if (m->handle) {
        /* TODO: format-specific teardown */
        m->handle = NULL;
    }
    free(m);
}
