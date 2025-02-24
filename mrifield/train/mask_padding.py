import torch

class SubjectMaskPadding:
    def __init__(self, padding: int = 1):
        self.padding = padding
        self.padding_filter = torch.ones([1,1] + [self.padding*2 + 1]*3, dtype=torch.float32).cuda()

    def __call__(self, input_shape_mask: torch.Tensor) -> torch.Tensor:
        check_border = torch.nn.functional.conv3d(input_shape_mask.type(torch.float32), self.padding_filter, padding=self.padding)
        return check_border == torch.sum(self.padding_filter)
    