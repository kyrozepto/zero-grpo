from setuptools import setup, find_packages

setup(
    name="zero-grpo",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "kornia",
        "opencv-python",
        "timm",
        "scipy",
        "numpy",
        "matplotlib",
        "seaborn",
        "pandas",
        "tqdm",
        "Pillow",
    ],
    author="Bahiskara Ananda Arryanto",
    python_requires=">=3.8",
)
