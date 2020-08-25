#include <torch/extension.h>
#include <vector>

int emd_forward_cuda(at::Tensor xyz1, at::Tensor xyz2, at::Tensor match, at::Tensor cost, at::Tensor temp);

int emd_backward_cuda(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz1, at::Tensor gradxyz2, at::Tensor match);

int emd_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor match, at::Tensor cost, at::Tensor temp){
	return emd_forward_cuda(xyz1, xyz2, match, cost, temp);
}

int emd_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz1, at::Tensor gradxyz2, at::Tensor match){
	return emd_backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, match);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &emd_forward, "emd forward (CUDA)");
  m.def("backward", &emd_backward, "emd backward (CUDA)");
}
