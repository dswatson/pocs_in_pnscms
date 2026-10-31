"""One-command reproduction of all frozen manuscript figures."""
from pathlib import Path
from experiments import applications,simulations


def main():
    root=Path(__file__).resolve().parent
    out=root/'results/figures'
    out.mkdir(parents=True,exist_ok=True)
    applications.main(out)
    simulations.main(out)
    print('\nAll manuscript figures generated in',out)


if __name__=='__main__':main()
