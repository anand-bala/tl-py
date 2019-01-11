# Usage

In this example, we are going to look at the basic ways to declare and use STL.

Begin by importing the STL specific subpackage

```pythonstub
import temporal_logic.signal_tl as stl
```

Declare the signals that are present in the trace, i.e., assign variables that represent the state space of a system. For convenience, we will declare it as a list or tuple

```pythonstub
signals = ('x', 'y', 'z')
x, y, z = stl.signals(signals) 
```

Not, the variables on the LHS can now be used as symbolic expressions, exactly like in [SymPy](https://www.sympy.org/en/index.html). This is because `x`, `y`, and `z` _are_ SymPy symbols!

Let us now write an STL property. Say, the following property

$$ \phi_1 = G ( (x > 0) \land (y < 0) ) $$

```python
phi1 = stl.G((x > 0) & (y > 0))
```

Woah! This looks almost exactly like the TeX expression of the formula.

This package has been fitted with a whole bunch of syntactical sugar to make your life easier. Internally the `phi1` would look like

```python
phi1 = stl.Globally(stl.And(stl.Predicate(x > 0), stl.Predicate(y > 0)))
```

Also, if you notice, the expression `phi1` is structured like a tree. This is because it is exactly that, and uses ideas from SymPy. So if you want to iterate over the nodes of the tree, just look at examples from [SymPy Advanced Expression Manipulation](https://docs.sympy.org/latest/tutorial/manipulation.html)

Moreover, to do pre-order iteration over the syntax tree you can simply do

```python

for arg in preorder_iterator(phi1):
    ...
```

## STL Semantics

In this package, we have defined multiple semantics for the STL grammar. These can be accessed through the `temporal_logic.signal_tl.semantics` package. For example, for accessing the boolean semantics and checking SAT, just use the `check_sat` function, which is defined as

```python
def check_sat(phi, signals, trace, t=None, dt=inf) -> bool
```

This function takes

- `phi`: The STL formula
- `signals`: The list of signals (like we defined at the top of this file)
- `trace`: The trace of the system
- `t`: (Optional) The timestamps corresponding to the samples in the trace
- `dt`: (Optional) The smallest time delta that the trace needs to be resampled at

And returns a `True` or `False` based on if the property is satisfied.

