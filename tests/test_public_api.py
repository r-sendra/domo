"""
Guard against __all__ / re-export drift: every name a package advertises in
__all__ must actually be importable from the package (not just its submodule).
This catches the class of bug where a symbol is added to a submodule but the
package __init__ forgets to re-export it — invisible to feature tests that
import from the submodule, but fatal to code (and examples) using the public
package path.
"""

import importlib

import pytest

PACKAGES = [
    "domo",
    "domo.control",
    "domo.robot",
    "domo.sim",
    "domo.tasks",
    "domo.skills",
    "domo.scenes",
    "domo.rl",
    "domo.llm",
    "domo.eureka",
]


@pytest.mark.parametrize("pkg_name", PACKAGES)
def test_all_names_importable(pkg_name):
    pkg = importlib.import_module(pkg_name)
    missing = [name for name in getattr(pkg, "__all__", [])
               if not hasattr(pkg, name)]
    assert not missing, f"{pkg_name}.__all__ advertises unresolved names: {missing}"


def test_demo_import_surface():
    # The exact symbols examples/eureka/eureka_getup.py::demo imports.
    from domo.control import LearnedJointSkill, SimControlLoop, SingleSkillController
    assert LearnedJointSkill and SimControlLoop and SingleSkillController
