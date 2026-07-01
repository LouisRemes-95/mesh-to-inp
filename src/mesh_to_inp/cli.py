import argparse
from pathlib import Path

from rich.console import Console
from rich.traceback import install

install(show_locals=True)

from mesh_to_inp.convert import convert
from mesh_to_inp.errors import UserError
from mesh_to_inp.config import load_case


console = Console()


class _RichArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        console.print(f"[bold red]Argument error:[/bold red] {message}\n")
        self.print_help()
        raise SystemExit(2)
    

def _build_parser() -> _RichArgumentParser:
    parser = _RichArgumentParser(
        prog = "mesh-to-inp-mesh", 
        description = "Convert a .mesh (or other meshio-supported mesh) to .inp with interfaces",
    )
        
    parser.add_argument(
        "case_path",
        type=Path,
        help="Path to the YAML case file",
    )

    return parser


def main():
    parser = _build_parser()

    console.print("[bold]mesh-to-inp-mesh[/bold]")

    args = parser.parse_args()

    try:
        case = load_case(args.case_path)

        with console.status("[cyan]Converting to .inp..."):
            convert(case)

        console.print(f"[green]✔ Converting to .inp complete[/green]")

        try:
            rel = case.job.output.relative_to(Path.cwd())
        except ValueError:
            rel = case.job.output

        console.print(f"Wrote: {rel}")
        console.print(f"Loaded {len(case.materials)} material(s)")
    
    except UserError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()