class Session:
    def __init__(self, *args, **kwargs):
        pass

    def post(self, *args, **kwargs):
        raise RuntimeError("requests stub: network operations are unavailable in tests")

    def put(self, *args, **kwargs):
        raise RuntimeError("requests stub: network operations are unavailable in tests")

    def mount(self, *args, **kwargs):
        return None


def __getattr__(name):
    raise RuntimeError(f"requests stub does not implement '{name}'")
