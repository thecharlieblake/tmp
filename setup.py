import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension

setuptools.setup(
    name="scale_op",
    description="scale_op",
    packages=["scale_op"],
    install_requires=["torch"],
    cmdclass={"build_ext": BuildExtension},
    ext_modules=[
        CppExtension(
            name="_scale", sources=["_scale/op.cpp"], extra_compile_args=["-std=c++14"]
        )
    ],
)
