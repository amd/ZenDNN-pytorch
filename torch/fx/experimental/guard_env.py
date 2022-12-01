
class GuardEnv:
    pass
# TODO(voz): This is super lame, but keeping it global for the sake of the prototype
# allows me to punt dealing with kwargs, piping stuff around backends, etc
# We may or may not want to land like this, TBD, but for now it allows me to focus
# on implementation. We got away with reading the fake_mode off of tensors, so maybe we
# will do the same here. Maybe not. 
GUARD_ENV = GuardEnv()

