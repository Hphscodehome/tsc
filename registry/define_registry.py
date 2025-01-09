#region other-package
from collections import defaultdict
#endregion


class Registry:
    mapping = defaultdict(dict)
    @classmethod
    def register(cls, domain, name):
        def wrap(f):
            cls.mapping[domain][name] = f
            return f
        return wrap

Registry()