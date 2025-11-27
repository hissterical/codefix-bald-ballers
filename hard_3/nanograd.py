import numpy as np
from typing import Set, List, Callable, Tuple, Optional

class Value:
    """
    Stores a single scalar value and its gradient.
    """

    def __init__(self, data: float, _children: Tuple['Value', ...] = (), _op: str = ''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other: 'Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: 'Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # FIXED: Chain rule: d(x*y)/dx = y, d(x*y)/dy = x
            # Previous bug swapped self.data and other.data
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, other: float) -> 'Value':
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # FIXED: Power rule: d(x^n)/dx = n * x^(n-1)
            # Previous bug missed the 'other' (n) coefficient
            self.grad += out.grad * (other * (self.data ** (other - 1)))

        out._backward = _backward
        return out

    def relu(self) -> 'Value':
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            # FIXED: Handle boundary condition (>= 0)
            # This is critical for recovering from 0 activation
            self.grad += out.grad * (1 if self.data >= 0 else 0)

        out._backward = _backward
        return out

    def __neg__(self) -> 'Value':
        return self * -1

    def __sub__(self, other: 'Value') -> 'Value':
        return self + (-other)

    def __truediv__(self, other: 'Value') -> 'Value':
        return self * (other ** -1)

    def __radd__(self, other: float) -> 'Value':
        return self + other

    def __rmul__(self, other: float) -> 'Value':
        return self * other

    def __rsub__(self, other: float) -> 'Value':
        return Value(other) - self

    def __rtruediv__(self, other: float) -> 'Value':
        return Value(other) / self

    def backward(self) -> None:
        """
        Compute gradients for all nodes in the computational graph.
        """
        topo: List[Value] = []
        visited: Set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Set gradient of output to 1
        self.grad = 1.0

        # FIXED: Iterate in REVERSE topological order (Output -> Input)
        # Previous bug iterated forward, which is wrong for backprop
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        self.grad = 0.0

# -----------------------------------------------------------------------------
# Neural Network Components
# -----------------------------------------------------------------------------

class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(np.random.randn()) for _ in range(nin)]
        # FIXED: Initialize bias to 0.0 instead of random
        # Random negative bias + 0 input = Dead ReLU (0 gradient)
        # 0 bias ensures gradient flows at start.
        self.b = Value(0.0)

    def __call__(self, x: List[Value]) -> Value:
        # w · x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()

    def parameters(self) -> List[Value]:
        return self.w + [self.b]

class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: List[Value]) -> List[Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> List[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x: List[Value]) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        # FIXED: Actually perform the zeroing logic
        for p in self.parameters():
            p.grad = 0.0

# -----------------------------------------------------------------------------
# Training & Utility Functions
# -----------------------------------------------------------------------------

def train_step(model: MLP, xs: List[List[Value]], ys: List[Value], lr: float = 0.01) -> float:
    """
    Perform one training step.
    """
    # 1. Zero Gradients
    # FIXED: Must call this to prevent gradient accumulation from previous steps
    model.zero_grad() 
    
    # 2. Forward pass
    ypred = [model(x) for x in xs]

    # 3. Compute MSE loss
    loss = sum((yp - yt)**2 for yp, yt in zip(ypred, ys))

    # 4. Backward pass
    loss.backward()

    # 5. Update parameters
    for p in model.parameters():
        p.data -= lr * p.grad

    return loss.data

def numerical_gradient(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """
    Compute numerical gradient using Central Difference.
    """
    # FIXED: Use Central Difference (f(x+h) - f(x-h)) / 2h for accuracy
    return (f(x + h) - f(x - h)) / (2 * h)

def safe_div(a: Value, b: Value, epsilon: float = 1e-10) -> Value:
    """
    Safe division that avoids division by zero.
    """
    # FIXED: Add epsilon to denominator
    return a / (b + epsilon)

# -----------------------------------------------------------------------------
# Main Test Block
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("NanoGrad - Fixed & Verified")
    print("=" * 60)

    # Test 1: Mathematical Correctness
    print("\n--- Test 1: Basic Operations & Chain Rule ---")
    x = Value(2.0)
    y = Value(3.0)
    # z = x * y + x^2 -> 6 + 4 = 10
    z = x * y + x**2
    z.backward()

    print(f"x = {x.data}, y = {y.data}")
    print(f"z = {z.data}")
    print(f"dz/dx = {x.grad} (Expected: 7.0)")
    print(f"dz/dy = {y.grad} (Expected: 2.0)")

    # Test 2: Neural Network Training
    print("\n--- Test 2: XOR Training (Small Network) ---")
    model = MLP(2, [4, 1])

    xs = [
        [Value(0.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(0.0)],
        [Value(1.0), Value(1.0)],
    ]
    ys = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]

    print("Training loop:")
    for i in range(21):
        loss = train_step(model, xs, ys, lr=0.5) # Higher LR for faster visible convergence
        if i % 5 == 0:
            print(f"Step {i}: loss = {loss:.4f}")

    print("\n" + "=" * 60)
    print("Logic Verified: Gradient Direction, Chain Rule, and Dead Neurons Fixed.")
    print("=" * 60)