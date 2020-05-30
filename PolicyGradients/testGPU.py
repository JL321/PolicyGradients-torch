import torch
import gym

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())

def demoGPU():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))
    x = torch.tensor([1, 2], dtype=torch.float32, requires_grad=True, device=device)
    y = torch.tensor([3, 4], dtype=torch.float32).to(device)
    a = x+y
    b = x*y
    c = (a+b).mean()
    c.backward()
    print(x.grad)
    print(c)
    env = gym.make('BipedalWalker-v3')
    print("Done Env1!")
    assert False

if __name__ == '__main__':
    demoGPU()
