#pragma once

// Command entry points. argv[0] is the command name; main() dispatches here.

int cmd_drift(int argc, char** argv);
int cmd_reduce(int argc, char** argv);
int cmd_bench(int argc, char** argv);
