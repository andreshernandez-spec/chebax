"""Base class for recipe callables, retiring per-class pytree boilerplate.

A recipe instance is a pytree whose children are ChebSeries (listed in
_series_fields) and whose auxiliary data are static scalars (listed in
_static_fields). Constructors take statics first, series after, in field
order; _post_init derives cached constants (called by both __init__ and
tree_unflatten). astype casts every series; __repr__ shows the statics and
the coefficient dtype.

Subclasses register themselves with the standard
@jax.tree_util.register_pytree_node_class decorator at the class site.
"""

class Recipe:
    _static_fields = ()
    _series_fields = ()

    def __init__(self, *args):
        names = self._static_fields + self._series_fields
        if len(args) != len(names):
            raise TypeError(f"{type(self).__name__} expects {names}, got {len(args)} args")
        for name, val in zip(names, args):
            setattr(self, name, val)
        self._post_init()

    def _post_init(self):
        pass

    def astype(self, dtype):
        return type(self)(*[getattr(self, f) for f in self._static_fields],
                          *[getattr(self, f).astype(dtype) for f in self._series_fields])

    def __repr__(self):
        statics = ", ".join(f"{f}={getattr(self, f)!r}" for f in self._static_fields)
        dtype = "-"
        # first series child with a coefficient array; children may be
        # nested recipes (spherical wraps a cylindrical instance)
        node = self
        while getattr(node, "_series_fields", ()):
            node = getattr(node, node._series_fields[0])
            coef = getattr(node, "coef", None)
            if coef is not None:
                dtype = coef.dtype
                break
        return f"{type(self).__name__}({statics}, dtype={dtype})"

    def tree_flatten(self):
        return (tuple(getattr(self, f) for f in self._series_fields),
                tuple(getattr(self, f) for f in self._static_fields))

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        for name, val in zip(cls._static_fields, aux):
            setattr(obj, name, val)
        for name, val in zip(cls._series_fields, children):
            setattr(obj, name, val)
        obj._post_init()
        return obj
