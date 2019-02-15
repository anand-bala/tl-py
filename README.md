# `temporal_logic` package for Python

This package provides an interface to defining and using Temporal Logics. Currently, Signal Temporal Logic grammar has been implemented along with a few monitors that check satisfiability of traces against a given STL specification.

This tool is inspired by Matlab toolboxes like [Breach](https://github.com/decyphir/breach) and [S-TaLiRo](https://sites.google.com/a/asu.edu/s-taliro/s-taliro), and Python toolboxes like [TuLiP](https://tulip-control.sourceforge.io/).


## Motivation

I got pretty annoyed with defining TL formulas as text (in files or as strings). So I thought I might as well write a package that provides an idiomatic interface to Temporal Logics. In this package, I aim to treat all operators and elements of a grammar as Python objects, hence allowing me to write formulas as Python expressions.

To do this, I heavily use [SymPy](https://www.sympy.org/en/index.html) and borrow many ideas from it.

# Installation

Using `pip`:

```
pip install git+https://github.com/anand-bala/tl-py@<VERSION>#egg=temporal_logic
```

Where you substitute `<VERSION>` with the desired version. As of writing, the latest is `v0.2.0`.


# Usage

In this example, we are going to look at the basic ways to declare and use STL.

Begin by importing the STL specific subpackage

```python
import temporal_logic.signal_tl as stl
```

Declare the signals that are present in the trace, i.e., assign variables that represent the state space of a system. For convenience, we will declare it as a list or tuple

```python
signals = ('x', 'y', 'z')
x, y, z = stl.signals(signals) 
```

Note: the variables on the LHS can now be used as symbolic expressions, exactly like in [SymPy](https://www.sympy.org/en/index.html). This is because `x`, `y`, and `z` _are_ SymPy symbols!

Let us now write an STL property. Say, the following property

<img src="https://latex.codecogs.com/svg.latex?\phi_1&space;=&space;\mathbf{G}&space;(&space;(x&space;>&space;0)&space;\land&space;(y&space;<&space;0)&space;)" title="\phi_1 = \mathbf{G} ( (x > 0) \land (y < 0) )" />

```python
phi1 = stl.G((x > 0) & (y > 0))
```

This package has been fitted with a whole bunch of syntactical sugar to make your life easier. Internally the `phi1` would look like

```python
phi1 = stl.Globally(stl.And(stl.Predicate(x > 0), stl.Predicate(y > 0)))
```
<!-- 

Also, if you notice, the expression `phi1` is structured like a tree. This is because it is exactly that, and uses ideas from SymPy. So if you want to iterate over the nodes of the tree, just look at examples from [SymPy Advanced Expression Manipulation](https://docs.sympy.org/latest/tutorial/manipulation.html)
 -->
Moreover, to do pre-order iteration over the syntax tree you can simply do

```python

for arg in preorder_iterator(phi1):
    ...
```

## STL Semantics

In this package, we have defined multiple semantics for the STL grammar. These can be accessed through the `temporal_logic.signal_tl.monitors` package. For example, for accessing the boolean semantics and checking SAT, just use the `eval_bool` function, which is defined as

```python
def eval_bool(phi: stl.Expression, w: pd.DataFrame, t=None) -> pd.Series:
```

This function takes

- `phi`: The STL formula
- `w`: The trace as a `pandas.DataFrame` compatible structure (including dict of `numpy` arrays)
- `t`: (Optional) The timestamps at which the satisfaction value need to be determined. This must be in the form of a list of indices/timestamps.

And returns a boolean array corresponding to each sample in the trace.


