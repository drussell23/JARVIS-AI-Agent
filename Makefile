# The bootstrap, in executable form.
#
# docs/BRINGUP_WINDOWS_WSL2.md describes standing this repo up as prose:
# create a venv, `pip install -e .` ("provides the `ov` script"), then run
# `ov doctor`. That sequence is correct and it has a gap, because it only
# works INSIDE an activated venv. A console entry point lives in the venv's
# own bin, and a login shell that never activates the venv cannot see it --
# so `ov` is "command not found" on a box where it is perfectly installed
# (observed 2026-09-05: the script sat at ~/.venvs/ov/bin/ov the whole time).
#
# `make link` closes that gap the way pipx does: a symlink from a directory
# already on PATH. The console script carries an ABSOLUTE shebang into its
# own interpreter, so it self-activates from wherever it is linked -- no
# wrapper, no PATH mutation, no profile edit.
#
#   make install   the full documented bootstrap: editable install + link
#   make link      expose ov/jarvis/trinity to a plain login shell
#   make verify    prove the entry points resolve and say what to fix
#   make unlink    remove the links this Makefile made
#
# Every path is resolved at run time. Nothing here assumes a home directory,
# a venv location, or a shell.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Where links go. `~/.local/bin` is the XDG-conventional user bin and is
# already on PATH in most shells; overridable for an unusual layout.
BINDIR ?= $(HOME)/.local/bin

# The console scripts this repo declares in [project.scripts]. Derived from
# pyproject rather than retyped, so adding an entry point there is enough.
ENTRY_POINTS := $(shell sed -n '/^\[project\.scripts\]/,/^\[/p' pyproject.toml \
	| grep -oE '^[a-zA-Z0-9_-]+[[:space:]]*=' | tr -d ' =' | tr '\n' ' ')

.PHONY: help install link unlink verify

help:
	@echo "bootstrap targets:"
	@echo "  make install   editable install + expose the entry points"
	@echo "  make link      expose $(ENTRY_POINTS)to a login shell"
	@echo "  make verify    prove they resolve; report what to fix"
	@echo "  make unlink    remove the links"
	@echo
	@echo "entry points (from pyproject.toml): $(ENTRY_POINTS)"
	@echo "link target (BINDIR):               $(BINDIR)"

install:
	@set -euo pipefail; \
	if [ -n "$${VIRTUAL_ENV:-}" ]; then \
	  echo "installing into the ACTIVE venv: $$VIRTUAL_ENV"; \
	  "$$VIRTUAL_ENV/bin/python" -m pip install -e . ; \
	else \
	  echo "no venv is active."; \
	  echo "Activate one first, or point OV_VENV at it:"; \
	  echo "    source .venv/bin/activate && make install"; \
	  echo "    OV_VENV=~/.venvs/ov make link      # already installed"; \
	  exit 2; \
	fi; \
	$(MAKE) --no-print-directory link

# --- the gap the prose bootstrap leaves -----------------------------------
# Resolution order, most explicit first. Each candidate is only accepted if
# it actually CONTAINS an entry point, so a stale or half-built venv is
# skipped rather than linked to.
link:
	@set -uo pipefail; \
	first="$$(echo $(ENTRY_POINTS) | awk '{print $$1}')"; \
	if [ -z "$$first" ]; then \
	  echo "REFUSING: pyproject.toml declares no [project.scripts]"; exit 2; \
	fi; \
	src=""; \
	if [ -n "$${OV_VENV:-}" ]; then \
	  if [ -x "$$OV_VENV/bin/$$first" ]; then \
	    src="$$OV_VENV/bin"; \
	  else \
	    echo "REFUSING: OV_VENV=$$OV_VENV does not contain '$$first'."; \
	    echo "  An explicit choice is honoured or refused, never quietly"; \
	    echo "  replaced with a different venv -- linking entry points from"; \
	    echo "  somewhere other than the one you named is how the wrong code"; \
	    echo "  ends up running under the right name."; \
	    exit 2; \
	  fi; \
	else \
	  for cand in "$${VIRTUAL_ENV:-}/bin" "$(CURDIR)/.venv/bin" \
	              "$(HOME)/.venvs/ov/bin"; do \
	    case "$$cand" in /bin) continue;; esac; \
	    if [ -x "$$cand/$$first" ]; then src="$$cand"; break; fi; \
	  done; \
	fi; \
	if [ -z "$$src" ]; then \
	  echo "REFUSING: found no venv containing '$$first'."; \
	  echo "  Looked in: \$$VIRTUAL_ENV/bin, ./.venv/bin, ~/.venvs/ov/bin"; \
	  echo "  Name one explicitly:  OV_VENV=~/.venvs/ov make link"; \
	  echo "  Or install first:     source <venv>/bin/activate && make install"; \
	  exit 2; \
	fi; \
	echo "entry points from: $$src"; \
	mkdir -p "$(BINDIR)" || { echo "REFUSING: cannot create $(BINDIR)"; exit 2; }; \
	linked=0; \
	for cmd in $(ENTRY_POINTS); do \
	  if [ -x "$$src/$$cmd" ]; then \
	    ln -sfn "$$src/$$cmd" "$(BINDIR)/$$cmd" \
	      && { echo "  linked $$cmd"; linked=$$((linked+1)); } \
	      || echo "  FAILED to link $$cmd"; \
	  else \
	    echo "  skipped $$cmd (not built in this venv)"; \
	  fi; \
	done; \
	echo "$$linked entry point(s) linked into $(BINDIR)"; \
	$(MAKE) --no-print-directory verify

# --- prove it, and say what to fix rather than fixing it silently ---------
# A profile is the user's file. Editing it automatically is how a bootstrap
# corrupts a config it does not understand, so this REPORTS and stops.
verify:
	@set -uo pipefail; \
	first="$$(echo $(ENTRY_POINTS) | awk '{print $$1}')"; \
	on_path=0; \
	case ":$$PATH:" in *":$(BINDIR):"*) on_path=1;; esac; \
	if [ "$$on_path" = "1" ]; then \
	  echo "PATH: $(BINDIR) is present"; \
	else \
	  case "$${SHELL:-}" in \
	    */zsh)  prof="$$HOME/.zshrc" ;; \
	    */bash) prof="$$HOME/.bashrc" ;; \
	    */fish) prof="$$HOME/.config/fish/config.fish" ;; \
	    *)      prof="your shell profile" ;; \
	  esac; \
	  echo "PATH: $(BINDIR) is NOT on PATH — the links exist but will not resolve."; \
	  echo "  Add this line to $$prof, then open a new shell:"; \
	  case "$${SHELL:-}" in \
	    */fish) echo "      fish_add_path $(BINDIR)" ;; \
	    *)      echo "      export PATH=\"$(BINDIR):\$$PATH\"" ;; \
	  esac; \
	fi; \
	if command -v "$$first" >/dev/null 2>&1; then \
	  echo "resolves: $$(command -v $$first)"; \
	  "$$first" version 2>/dev/null | head -1 || true; \
	else \
	  echo "does NOT resolve in this shell: $$first"; \
	  echo "  (a login shell may still see it — try: bash -lc 'command -v $$first')"; \
	fi; \
	$(MAKE) --no-print-directory verify-surface

# Resolving is not running. `ov` resolved perfectly on a box where the
# interactive cockpit could not start, because `pip install -e . --no-deps`
# installs no dependencies and nothing then checked that the cockpit's own
# imports were satisfied. The operator got a frozen crest and no prompt.
#
# The interpreter probed is the one the ENTRY POINT resolves through, read
# from its shebang — not `python3`, and not whatever venv happens to be
# active. That distinction IS the bug: the package was importable in the
# shell and missing from the venv `ov` actually runs in.
#
# Never fails the build. This target reports; the operator decides.
.PHONY: verify-surface
verify-surface:
	@set -uo pipefail; \
	first="$$(echo $(ENTRY_POINTS) | awk '{print $$1}')"; \
	bin="$$(command -v $$first 2>/dev/null || true)"; \
	if [ -z "$$bin" ]; then exit 0; fi; \
	py="$$(head -1 "$$bin" | sed -n 's|^#!\(.*\)$$|\1|p' | awk '{print $$1}')"; \
	if [ ! -x "$$py" ]; then py="$$(command -v python3 || true)"; fi; \
	if [ -z "$$py" ]; then echo "cockpit: no interpreter to probe"; exit 0; fi; \
	"$$py" -c "import sys; sys.path.insert(0, '.'); \
from backend.core.ouroboros.cli.surface_probe import probe_interactive_surface as p; \
v = p(stdin_isatty=True); \
print('cockpit: ready' if v.ok else 'cockpit: NOT ready — ' + v.reason); \
print('  fix: ' + v.remedy) if v.remedy else None" 2>/dev/null \
	  || echo "cockpit: could not probe (is this the repo root?)"

unlink:
	@set -uo pipefail; \
	for cmd in $(ENTRY_POINTS); do \
	  target="$(BINDIR)/$$cmd"; \
	  if [ -L "$$target" ]; then rm -f "$$target" && echo "  removed $$cmd"; \
	  elif [ -e "$$target" ]; then echo "  left $$cmd alone (not a symlink)"; \
	  fi; \
	done
