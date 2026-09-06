"""Render a Typer app as real Sphinx ``option`` objects.

Neither of the obvious off-the-shelf options works here:

- ``sphinx-click`` can't read a Typer app at all. typer >= 0.26 vendors its
  own click fork, so ``TyperGroup`` is an instance of
  ``typer._click.core.Command``, not the ``click.Command`` sphinx-click
  type-checks against.
- ``sphinxcontrib-typer`` works, but only ever emits a captured *terminal*
  rendering of ``--help`` -- an SVG screenshot, an HTML transcript, or a
  fixed-width text block. None of those are Sphinx objects: no per-option
  anchors, no index entries, no ``:option:`` cross-references, and a hard
  column width that scrolls sideways instead of reflowing.

So this walks the command tree itself and generates ``.. program::`` /
``.. option::`` directives. The content still comes entirely from the live
Typer definitions, so it can't drift from what ``--help`` prints.

Usage::

    .. typer-reference:: depth_anything_3.cli:app
       :prog: da3
"""

from __future__ import annotations

from importlib import import_module
from typing import Any
from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective


def _import_app(path: str) -> Any:
    """Import ``module:attr`` (or ``module.attr``) and return the object."""
    module_path, _, attr = path.partition(":")
    if not attr:
        module_path, _, attr = path.rpartition(".")
    return getattr(import_module(module_path), attr)


def _normalize_help(text: str) -> list[str]:
    """Make a docstring safe to parse as reStructuredText.

    Typer help text is written for a terminal, where a bullet list can follow
    its lead-in line directly. RST needs a blank line before the list, and the
    ``-W`` build turns the resulting warning into an error.
    """
    lines = text.strip().splitlines()
    out: list[str] = []
    for line in lines:
        is_bullet = line.lstrip().startswith(("- ", "* "))
        if is_bullet and out and out[-1].strip() and not out[-1].lstrip().startswith(("- ", "* ")):
            out.append("")
        out.append(line.rstrip())
    return out


def _metavar(param: Any) -> str:
    """The ``<value>`` placeholder to show after an option, if it takes one.

    Uses the parameter type's own name (``str``, ``int``, ``float``, ...) so
    the signature reads exactly as ``--help`` prints it.
    """
    if getattr(param, "is_flag", False):
        return ""
    if param.metavar:
        return f"<{param.metavar.lower()}>"
    choices = getattr(param.type, "choices", None)
    if choices:
        return "<" + "|".join(str(c) for c in choices) + ">"
    return f"<{getattr(param.type, 'name', 'value')}>"


def _signature(param: Any) -> str:
    """The ``.. option::`` signature for one parameter."""
    if param.param_type_name == "argument":
        return param.name.upper()
    metavar = _metavar(param)
    opts = list(param.opts) + list(param.secondary_opts)
    joined = ", ".join(opts)
    return f"{joined} {metavar}".strip() if metavar else joined


def _usage(prog: str, command: Any, is_group: bool) -> str:
    parts = [prog]
    if [p for p in command.params if p.param_type_name == "option"]:
        parts.append("[OPTIONS]")
    if is_group:
        parts += ["COMMAND", "[ARGS]..."]
    for param in command.params:
        if param.param_type_name != "argument":
            continue
        name = param.name.upper()
        if param.nargs == -1:
            name = f"{name}..."
        parts.append(name if param.required else f"[{name}]")
    return " ".join(parts)


def _literal(value: Any) -> str:
    """``value`` as an inline literal, degrading to plain text if it can't be.

    A backtick inside an inline literal has no escape in RST, so anything
    containing one is written unmarked rather than emitting broken markup.
    """
    text = str(value)
    return text if "`" in text else f"``{text}``"


def _default_note(param: Any) -> str | None:
    """``Default: ...`` for a parameter, or None when there's nothing to say.

    An empty string is how Typer spells "unset" for these options, and an
    empty inline literal (````) is both meaningless and invalid RST.
    """
    if param.param_type_name != "option":
        return None
    default = param.default
    if default is None or default == "" or default == () or default == [] or default == {}:
        return None
    return f"Default: {_literal(default)}."


def _param_rst(param: Any) -> list[str]:
    """One ``.. option::`` block: signature, help text, then default/required."""
    lines = [f".. option:: {_signature(param)}", ""]
    body: list[str] = []
    if param.help:
        body += _normalize_help(param.help)

    # Flag pairs already show both spellings in the signature, so the default
    # is the only thing left to state.
    note = "**Required.**" if param.required else _default_note(param)
    if note:
        body += ["", note]

    lines += [f"   {line}" if line else "" for line in body]
    lines.append("")
    return lines


def _command_rst(prog: str, command: Any, is_group: bool) -> list[str]:
    """The body of one command's section, as reStructuredText."""
    lines: list[str] = []
    help_text = command.help or command.short_help or ""
    if help_text:
        lines += _normalize_help(help_text) + [""]

    lines += [".. code-block:: console", "", f"   $ {_usage(prog, command, is_group)}", ""]
    lines += [f".. program:: {prog}", ""]

    arguments = [p for p in command.params if p.param_type_name == "argument" and not p.hidden]
    options = [p for p in command.params if p.param_type_name == "option" and not p.hidden]

    for title, params in (("Arguments", arguments), ("Options", options)):
        if not params:
            continue
        lines += [f".. rubric:: {title}", ""]
        for param in params:
            lines += _param_rst(param)
    return lines


class TyperReferenceDirective(SphinxDirective):
    """Generate a section per command, each with real ``option`` objects."""

    has_content = False
    required_arguments = 1
    option_spec = {"prog": directives.unchanged_required}

    def run(self) -> list[nodes.Node]:
        from typer.main import get_command

        app = _import_app(self.arguments[0])
        prog = self.options["prog"]
        return [self._section(prog, get_command(app), title=prog)]

    def _section(self, prog: str, command: Any, title: str) -> nodes.section:
        subcommands = getattr(command, "commands", {})
        section_id = nodes.make_id(prog)
        section = nodes.section(
            "",
            nodes.title(text=title),
            ids=[section_id],
            names=[nodes.fully_normalize_name(title)],
        )
        self.state.document.note_implicit_target(section, section)

        content = StringList(
            _command_rst(prog, command, is_group=bool(subcommands)),
            source=self.get_source_info()[0],
        )
        self.state.nested_parse(content, self.content_offset, section)

        for name, subcommand in subcommands.items():
            if subcommand.hidden:
                continue
            section += self._section(f"{prog} {name}", subcommand, title=name)
        return section


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_directive("typer-reference", TyperReferenceDirective)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
