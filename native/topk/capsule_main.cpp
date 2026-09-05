#include "capsule_transport.hpp"
#ifndef STRIXLAB_TOPK_TEST_FAULT
#define STRIXLAB_TOPK_TEST_FAULT none
#endif
int main(int argc, char **argv) {
  return strixlab::topk::capsule_main(
      argc, argv,
      strixlab::topk::host_fixture::Fault::STRIXLAB_TOPK_TEST_FAULT);
}
