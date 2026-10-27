/* Test executable only: use the selected interpreter's headers and PyConfig ABI. */
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>

static void checked(PyStatus status) {
    if (PyStatus_Exception(status)) Py_ExitStatusException(status);
}

static void setting(PyConfig *config, wchar_t **field, const char *name) {
    const char *value = getenv(name);
    if (!value) {
        fprintf(stderr, "missing interpreter configuration: %s\n", name);
        exit(1);
    }
    checked(PyConfig_SetBytesString(config, field, value));
}

void budget_initialize_python(void) {
    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);
    config.site_import = 0;
    config.parse_argv = 0;
    config.install_signal_handlers = 0;
    setting(&config, &config.program_name, "VOLAS_BUDGET_EXECUTABLE");
    setting(&config, &config.executable, "VOLAS_BUDGET_EXECUTABLE");
    setting(&config, &config.prefix, "VOLAS_BUDGET_PREFIX");
    setting(&config, &config.exec_prefix, "VOLAS_BUDGET_EXEC_PREFIX");
    setting(&config, &config.base_prefix, "VOLAS_BUDGET_BASE_PREFIX");
    setting(&config, &config.base_exec_prefix, "VOLAS_BUDGET_BASE_EXEC_PREFIX");
    const char *count = getenv("VOLAS_BUDGET_PATH_COUNT");
    if (!count) exit(1);
    for (int i = 0; i < atoi(count); ++i) {
        char name[64];
        snprintf(name, sizeof(name), "VOLAS_BUDGET_PATH_%d", i);
        wchar_t *path = NULL;
        setting(&config, &path, name);
        checked(PyWideStringList_Append(&config.module_search_paths, path));
        PyMem_RawFree(path);
    }
    config.module_search_paths_set = 1;
    checked(Py_InitializeFromConfig(&config));
    PyConfig_Clear(&config);
    PyEval_SaveThread();
}
