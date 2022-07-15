import os
import subprocess
import sys
from distutils.command.build_ext import build_ext  # type: ignore
from distutils.command.install import install as distutils_install  # type: ignore

from setuptools import Extension, setup
from wheel.bdist_wheel import bdist_wheel

ROOT = os.path.realpath(os.path.dirname(__file__))

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")


EQUISTORE_BUILD_TYPE = os.environ.get("EQUISTORE_BUILD_TYPE", "release")
if EQUISTORE_BUILD_TYPE not in ["debug", "release"]:
    raise Exception(
        f"invalid build type passed: '{EQUISTORE_BUILD_TYPE}', "
        "expected 'debug' or 'release'"
    )

RUST_BUILD_TARGET = os.environ.get("RUST_BUILD_TARGET", None)


class universal_wheel(bdist_wheel):
    # When building the wheel, the `wheel` package assumes that if we have a
    # binary extension then we are linking to `libpython.so`; and thus the wheel
    # is only usable with a single python version. This is not the case for
    # here, and the wheel will be compatible with any Python >=3.6. This is
    # tracked in https://github.com/pypa/wheel/issues/185, but until then we
    # manually override the wheel tag.
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        # tag[2:] contains the os/arch tags, we want to keep them
        return ("py3", "none") + tag[2:]


class cmake_ext(build_ext):
    """
    Build the native library using cmake
    """

    def run(self):
        source_dir = ROOT
        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(os.path.realpath(self.build_lib), "equistore")

        try:
            os.mkdir(build_dir)
        except OSError:
            pass

        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_BUILD_TYPE={EQUISTORE_BUILD_TYPE}",
            "-DBUILD_SHARED_LIBS=ON",
            "-DEQUISTORE_BUILD_FOR_PYTHON=ON",
            "-DEQUISTORE_SERIALIZATION=ON",
        ]

        if RUST_BUILD_TARGET is not None:
            cmake_options.append(f"-DRUST_BUILD_TARGET={RUST_BUILD_TARGET}")

        if "CARGO" in os.environ:
            cmake_options.append(f"-DCARGO_EXE={os.environ['CARGO']}")

        subprocess.run(
            ["cmake", source_dir, *cmake_options],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", build_dir, "--target", "install"],
            check=True,
        )


def get_version():
    """
    Get the version of equistore from the Cargo.toml file and git metadata.

    If git is available, it is used to check if we are installing a development
    version or a released version (by checking how many commits happened since
    the last tag).
    """

    # read version from Cargo.toml
    with open("Cargo.toml") as fd:
        for line in fd:
            if line.startswith("version"):
                _, version = line.split(" = ")
                # remove quotes
                version = version[1:-2]
                # take the first version in the file, this should be the right
                # version
                break

    # Add pre-release info the version
    try:
        tags_list = subprocess.run(
            ["git", "tag"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            check=True,
        )
        tags_list = tags_list.stdout.decode("utf8").strip()

        if tags_list == "":
            first_commit = subprocess.run(
                ["git", "rev-list", "--max-parents=0", "HEAD"],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                check=True,
            )
            reference = first_commit.stdout.decode("utf8").strip()

        else:
            last_tag = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                check=True,
            )

            reference = last_tag.stdout.decode("utf8").strip()

    except Exception:
        reference = ""
        pass

    try:
        n_commits_since_tag = subprocess.run(
            ["git", "rev-list", f"{reference}..HEAD", "--count"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            check=True,
        )
        n_commits_since_tag = n_commits_since_tag.stdout.decode("utf8").strip()

        if n_commits_since_tag != 0:
            version += ".dev" + n_commits_since_tag
    except Exception:
        pass

    return version


setup(
    version=get_version(),
    ext_modules=[
        # only declare the extension, it is built & copied as required by cmake
        # in the build_ext command
        Extension(name="equistore", sources=[]),
    ],
    cmdclass={
        "build_ext": cmake_ext,
        "bdist_wheel": universal_wheel,
        # HACK: do not use the new setuptools install implementation, it tries
        # to install the package with `easy_install`, which fails to resolve the
        # freshly installed package and tries to load it from pypi.
        "install": distutils_install,
    },
    package_data={
        "equistore": [
            "equistore/lib/*",
            "equistore/include/*",
        ]
    },
)
