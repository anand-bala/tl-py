# `temporal_logic` package for Python

This package provides an interface to defining and using Temporal Logics. Currently,
Signal Temporal Logic grammar has been implemented along with a few monitors that check
satisfiability of traces against a given STL specification.

This tool is inspired by Matlab toolboxes like
[Breach](https://github.com/decyphir/breach) and
[S-TaLiRo](https://sites.google.com/a/asu.edu/s-taliro/s-taliro), and Python toolboxes
like [TuLiP](https://tulip-control.sourceforge.io/).


## Motivation

I got pretty annoyed with defining TL formulas as text (in files or as strings). So I
thought I might as well write a package that provides an idiomatic interface to Temporal
Logics. In this package, I aim to treat all operators and elements of a grammar as
Python objects, hence allowing me to write formulas as Python expressions.

To do this, I heavily use [SymPy](https://www.sympy.org/en/index.html) and borrow many ideas from it.

## Installation

*Recommended* to use [conda](https://docs.conda.io/en/latest/) to manage the
environment. If using `conda`, run the following in an active environment first:

```shell
$ conda install numba numpy scipy
$ conda install -c conda-forge sympy
```

Then (this is for anyone who doesn't want to clone the project) using `pip`:

```shell
$ pip install git+https://github.com/anand-bala/tl-py@<branch>#egg=temporal-logic
```
where `<branch>` should be replaced with the git branch, tag, or commit of your choice.

## Related projects

- [py-metric-temporal-logic](https://github.com/mvcisback/py-metric-temporal-logic.git)
    - This project is really good, but I wanted to use different quantitative
      semantics defined for STL, for example, [filtering semantics](https://arxiv.org/pdf/1510.08079.pdf).

