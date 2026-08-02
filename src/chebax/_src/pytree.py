"""Base class for recipe callables, retiring per-class pytree boilerplate.

A recipe instance is a pytree whose children are ChebSeries (listed in
_series_fields) and whose auxiliary data are static scalars (listed in
_static_fields). Constructors take statics first, series after, in field
order; _post_init derives cached constants (called by both __init__ and
tree_unflatten). astype casts every series; __repr__ shows the statics and
the coefficient dtype.

Subclasses register themselves with the standard
@jax.tree_util.register_pytree_node_class decorator at the class site.

Instances are frozen once _post_init returns. The factories hand the SAME
object to every caller asking for the same parameters (they are lru_cached),
so a caller mutating one would corrupt every later call served from that
slot; build a new instance instead.
"""

class Recipe:
    _static_fields = ()
    _series_fields = ()
    _frozen = False

    def __init__(self, *args):
        names = self._static_fields + self._series_fields
        if len(args) != len(names):
            raise TypeError(f"{type(self).__name__} expects {names}, got {len(args)} args")
        for name, val in zip(names, args):
            setattr(self, name, val)
        self._post_init()
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if self._frozen:
            raise AttributeError(
                f"{type(self).__name__} is immutable (cached instances are shared); "
                f"cannot set {name!r}")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if self._frozen:
            raise AttributeError(
                f"{type(self).__name__} is immutable (cached instances are shared); "
                f"cannot delete {name!r}")
        object.__delattr__(self, name)

    def _post_init(self):
        pass

    def astype(self, dtype):
        return type(self)(*[getattr(self, f) for f in self._static_fields],
                          *[getattr(self, f).astype(dtype) for f in self._series_fields])

    def truncate(self, tol):
        """Instance with every series' converged tail dropped (see
        ChebSeries.truncate); nested recipe children truncate too."""
        return type(self)(*[getattr(self, f) for f in self._static_fields],
                          *[getattr(self, f).truncate(tol) for f in self._series_fields])

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
        object.__setattr__(obj, "_frozen", True)
        return obj
