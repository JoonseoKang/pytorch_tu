import numpy as np

N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    loss =  np.square(y_pred -y).sum()
    print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 =x.T.dot(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

##########################torch tensor#############################

import torch

dtype = torch.float
device = torch.device('cuda')

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(H, D_out,device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    h = x.mm(w1)
    h_relu = h.clamp(min = 0)
    y_pred = h_relu.mm(w2)

    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 ==99:
        print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[ h<0 ] = 0
    grad_w1 = x.t().mm(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

#################autograd##################

import torch

dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거하세요.

# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉층의 차원이며, D_out은 출력 차원입니다.
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
# requires_grad=False로 설정하여 역전파 중에 이 Tensor들에 대한 변화도를 계산할
# 필요가 없음을 나타냅니다. (requres_grad의 기본값이 False이므로 아래 코드에는
# 이를 반영하지 않았습니다.)
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 가중치를 저장하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
# requires_grad=True로 설정하여 역전파 중에 이 Tensor들에 대한
# 변화도를 계산할 필요가 있음을 나타냅니다.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 순전파 단계: Tensor 연산을 사용하여 예상되는 y 값을 계산합니다. 이는 Tensor를
    # 사용한 순전파 단계와 완전히 동일하지만, 역전파 단계를 별도로 구현하지 않아도
    # 되므로 중간값들에 대한 참조(reference)를 갖고 있을 필요가 없습니다.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Tensor 연산을 사용하여 손실을 계산하고 출력합니다.
    # loss는 (1,) 형태의 Tensor이며, loss.item()은 loss의 스칼라 값입니다.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # autograd를 사용하여 역전파 단계를 계산합니다. 이는 requires_grad=True를
    # 갖는 모든 Tensor에 대해 손실의 변화도를 계산합니다. 이후 w1.grad와 w2.grad는
    # w1과 w2 각각에 대한 손실의 변화도를 갖는 Tensor가 됩니다.
    loss.backward()

    # 경사하강법(gradient descent)을 사용하여 가중치를 수동으로 갱신합니다.
    # torch.no_grad()로 감싸는 이유는 가중치들이 requires_grad=True이지만
    # autograd에서는 이를 추적할 필요가 없기 때문입니다.
    # 다른 방법은 weight.data 및 weight.grad.data를 조작하는 방법입니다.
    # tensor.data가 tensor의 저장공간을 공유하기는 하지만, 이력을
    # 추적하지 않는다는 것을 기억하십시오.
    # 또한, 이를 위해 torch.optim.SGD 를 사용할 수도 있습니다.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 가중치 갱신 후에는 수동으로 변화도를 0으로 만듭니다.
        w1.grad.zero_()
        w2.grad.zero_()



# PyTorch: 새 autograd 함수 정의하기
#
# 내부적으로, autograd의 기본(primitive) 연산자는 실제로 Tensor를 조작하는 2개의 함수입니다.
# forward 함수는 입력 Tensor로부터 출력 Tensor를 계산합니다. backward 함수는 어떤 스칼라 값에 대한 출력 Tensor의 변화도를 전달받고,
# 동일한 스칼라 값에 대한 입력 Tensor의 변화도를 계산합니다.
# PyTorch에서 torch.autograd.Function 의 서브클래스(subclass)를 정의하고
# forward 와 backward 함수를 구현함으로써 사용자 정의 autograd 연산자를 손쉽게 정의할 수 있습니다.
# 그 후, 인스턴스(instance)를 생성하고 이를 함수처럼 호출하여 입력 데이터를 갖는 Tensor를 전달하는 식으로
# 새로운 autograd 연산자를 사용할 수 있습니다.
# 이 예제에서는 ReLU로 비선형적(nonlinearity)으로 동작하는 사용자 정의 autograd 함수를 정의하고
# 2-계층 신경망에 이를 적용해보도록 하겠습니다:

import torch

class MyReLU(torch.autograd.Function):

    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        return grad_input

dtype = torch.float
device = torch.device('cuda')

N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 가중치를 저장하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 사용자 정의 Function을 적용하기 위해 Function.apply 메소드를 사용합니다.
    # 여기에 'relu'라는 이름을 붙였습니다.
    relu = MyReLU.apply

    # 순전파 단계: Tensor 연산을 사용하여 예상되는 y 값을 계산합니다;
    # 사용자 정의 autograd 연산을 사용하여 ReLU를 계산합니다.
    y_pred = relu(x.mm(w1)).mm(w2)

    # 손실을 계산하고 출력합니다.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # autograde를 사용하여 역전파 단계를 계산합니다.
    loss.backward()

    # 경사하강법(gradient descent)을 사용하여 가중치를 갱신합니다.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 가중치 갱신 후에는 수동으로 변화도를 0으로 만듭니다.
        w1.grad.zero_()
        w2.grad.zero_()


# -*- coding: utf-8 -*-
import torch

# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉층의 차원이며, D_out은 출력 차원입니다.
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# nn 패키지를 사용하여 모델을 순차적 계층(sequence of layers)으로 정의합니다.
# nn.Sequential은 다른 Module들을 포함하는 Module로, 그 Module들을 순차적으로
# 적용하여 출력을 생성합니다. 각각의 Linear Module은 선형 함수를 사용하여
# 입력으로부터 출력을 계산하고, 내부 Tensor에 가중치와 편향을 저장합니다.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# 또한 nn 패키지에는 널리 사용하는 손실 함수들에 대한 정의도 포함하고 있습니다;
# 여기에서는 평균 제곱 오차(MSE; Mean Squared Error)를 손실 함수로 사용하겠습니다.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # 순전파 단계: 모델에 x를 전달하여 예상되는 y 값을 계산합니다. Module 객체는
    # __call__ 연산자를 덮어써(override) 함수처럼 호출할 수 있게 합니다.
    # 이렇게 함으로써 입력 데이터의 Tensor를 Module에 전달하여 출력 데이터의
    # Tensor를 생성합니다.
    y_pred = model(x)

    # 손실을 계산하고 출력합니다. 예측한 y와 정답인 y를 갖는 Tensor들을 전달하고,
    # 손실 함수는 손실 값을 갖는 Tensor를 반환합니다.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 역전파 단계를 실행하기 전에 변화도를 0으로 만듭니다.
    model.zero_grad()

    # 역전파 단계: 모델의 학습 가능한 모든 매개변수에 대해 손실의 변화도를
    # 계산합니다. 내부적으로 각 Module의 매개변수는 requires_grad=True 일 때
    # Tensor 내에 저장되므로, 이 호출은 모든 모델의 모든 학습 가능한 매개변수의
    # 변화도를 계산하게 됩니다.
    loss.backward()

    # 경사하강법(gradient descent)를 사용하여 가중치를 갱신합니다. 각 매개변수는
    # Tensor이므로 이전에 했던 것과 같이 변화도에 접근할 수 있습니다.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad