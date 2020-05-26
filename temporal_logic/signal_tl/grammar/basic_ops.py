from .expression import Expression, as_Expression


class LogicOp(Expression):
    is_LogicOp = True

    @classmethod
    def _filter_args(cls, *args) -> tuple:
        return tuple(map(as_Expression, args))


class Not(LogicOp):
    """The Not operator.

    The Not operator contains a single operand, which can be an STL expression.
    """

    nargs = 1

    def tex_print(self):
        return r" \neg ({}) ".format(self.args[0].tex_print())


class Or(LogicOp):
    """The Or operator

    This is a n-ary operator where n >= 2.
    """

    nargs = None

    def tex_print(self):
        tex_args = tuple(arg.tex_print() for arg in self.args)
        return r" \lor ".join(tex_args)


class And(LogicOp):
    """The And operator

    This is a n-ary operator where n >= 2.
    """

    nargs = None

    def tex_print(self):
        tex_args = tuple(arg.tex_print() for arg in self.args)
        return r" \land ".join(tex_args)


class Implies(LogicOp):
    nargs = 2

    def __new__(cls, *args, **kwargs):
        if cls.nargs != len(args):
            raise ValueError("Incompatible number of args")

        p, q = tuple(map(as_Expression, args))

        return Or(Not(p), q)

    def tex_print(self):
        pass


__all__ = [
    "And",
    "Or",
    "Not",
    "Implies",
]
