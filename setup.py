from setuptools import setup

setup(
    name="lfcppcuda",
    version="0.1",
    packages=['lffastlib'],  # Explicitly list packages
    package_dir={'': '.'},  # Look in current directory
    install_requires=[
        "snorkel>=0.9.8",    # Or your required version
        #"pycuda>=2022.2.2",  # Specify a version as needed
        # Add other dependencies here (e.g., numpy, pandas, etc.)
    ],
    python_requires=">=3.6",  # Adjust Python version as needed
)
