# Backend-aware install targets.
#
# torch isn't in pyproject extras because pip can't pin an --index-url per
# extra (see pyproject.toml). scripts/install_backend.sh does the right
# thing per backend; these targets are thin aliases.

.PHONY: install install-rocm install-rocm6.4 install-rocm7.0 install-cuda install-cpu \
	install-vision-rocm install-vision-rocm6.4 install-vision-rocm7.0 \
	install-vision-cuda install-vision-cpu

install: install-rocm  ## default: ROCm (most AMD AI hosts)

install-rocm:
	./scripts/install_backend.sh rocm

install-rocm6.4:
	./scripts/install_backend.sh rocm6.4

install-rocm7.0:
	./scripts/install_backend.sh rocm7.0

install-cuda:
	./scripts/install_backend.sh cuda

install-cpu:
	./scripts/install_backend.sh cpu

install-vision-rocm:
	./scripts/install_backend.sh rocm vision

install-vision-rocm6.4:
	./scripts/install_backend.sh rocm6.4 vision

install-vision-rocm7.0:
	./scripts/install_backend.sh rocm7.0 vision

install-vision-cuda:
	./scripts/install_backend.sh cuda vision

install-vision-cpu:
	./scripts/install_backend.sh cpu vision
