import unittest

from .ignore_components_residual import *

class TestComponentResidual(unittest.TestCase):
    def test_dynamics(self):
        component = component_representation_residual(4, 4)
   #     m = nn.Conv2d(16, 1, 3)
  #      input = torch.randn(1, 1, 50, 16)
 #       output = m(input).view(-1, 16 * 50)
#        print(output)
#        assert not torch.is_tensor(output)
        img = torch.rand(1, 1, 3, 3)
        model = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=1, padding=0, bias=False)
        res = model(img).view(1, 4)
        print(res)

        assert not torch.is_tensor(res)

        #print(component)
        #print(component.preprocess_state)
        ##output = component(torch.tensor([[
       #     [0, 0, 0, 0],
       #     [0, 1, 1, 0],
      #      [0, 0, 1, 0],
       #     [0, 0, 0, 0],
      #  ]]).reshape((1, 4, 4, 1)).float())
      #  assert torch.is_tensor(output)

        #action_size = 4
       # component2 = component_predictions_residual(4, action_size)
       # P, v = component2(output)
      #  assert P.shape[-1] == action_size
     #   assert v.shape[-1] == 1
    #    assert torch.is_tensor(v)
   #     assert torch.is_tensor(P)

        """
        component3 = component_dynamics_residual(4)
        new_state = component3(
            output,
            torch.tensor([4, 4, 4, 4]).reshape((4, 1, 1, 1))
        )
        assert not torch.is_tensor(new_state)
        """
if __name__ == "__main__":
    TestComponentResidual()