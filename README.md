# `temporal_logic` package for Python

This package provides an interface to defining and using Temporal Logics. Currently, Signal Temporal Logic grammar has been implemented along with a few monitors that check satisfiability of traces against a given STL specification.

This tool is inspired by Matlab toolboxes like [Breach](https://github.com/decyphir/breach) and [S-TaLiRo](https://sites.google.com/a/asu.edu/s-taliro/s-taliro), and Python toolboxes like [TuLiP](https://tulip-control.sourceforge.io/).


## Motivation

I got pretty annoyed with defining TL formulas as text (in files or as strings). So I thought I might as well write a package that provides an idiomatic interface to Temporal Logics. In this package, I aim to treat all operators and elements of a grammar as Python objects, hence allowing me to write formulas as Python expressions.

To do this, I heavily use [SymPy](https://www.sympy.org/en/index.html) and borrow many ideas from it.

# Installation

Using `pip`:

```
pip install git+https://github.com/anand-bala/tl-py@v0.1.0#egg=temporal_logic
```