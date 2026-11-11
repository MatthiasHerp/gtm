#!/usr/bin/env python3

from importlib.metadata import distributions

def main():
    packages = sorted(
        ((dist.metadata["Name"], dist.version) for dist in distributions()),
        key=lambda x: x[0].lower()
    )

    print(f"{'Package':<40}Version")
    print("-" * 60)
    for name, version in packages:
        print(f"{name:<40}{version}")

if __name__ == "__main__":
    main()