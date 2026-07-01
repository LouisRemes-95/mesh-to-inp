import argparse
from pathlib import Path

from rich.console import Console
from rich.traceback import install

install(show_locals=True)

from mesh_to_inp.convert import convert
from mesh_to_inp.errors import UserError


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
        "input_path",
        type = Path,
        help = "Path to the input mesh file",
        )
    parser.add_argument(
        "-o",
        "--out",
        type = Path,
        default = None,
        help = "Path where the output needs to be saved (with file name and suffix)"
    )

    return parser


def _resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    return (output_path or input_path.with_suffix(".inp")).resolve()


def main():
    parser = _build_parser()

    console.print("[bold]mesh-to-inp-mesh[/bold]")

    args = parser.parse_args()
    output_path =  _resolve_output_path(args.input_path, args.out)

    try:
        with console.status("[cyan]Converting to .inp..."):
            convert(args.input_path, output_path)
        console.print(f"[green]✔ Converting to .inp complete[/green]")
    
        try:
            rel = output_path.relative_to(Path.cwd())
        except ValueError:
            rel = output_path

        console.print(f"Wrote: {rel}")
    
    except UserError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()