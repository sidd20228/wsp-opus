"""Experiment 8 rebuild (Qwen3.5-0.8B, thinking on, math by weight editing): the editor's environment and the evaluation.

  kernel.Kernel(run_id).run(code) -> str          the persistent CPU Python kernel (Modal Sandbox, no network)
  fwd.forward_pass(request_json, k, run_id, kernel=None) -> str   one forward-pass request; JSON text, "ok" false = failed
  fwd.pass_message(result_text, k, n_done, budget) -> str         the original harness's wording of a pass result
  evaluate.evaluate(source, sets, name) -> dict    benchmark evaluation (released evaluate.py protocol think35)
  phase0.py                                        Phase 0 check 0.7
  app.py                                           the Modal app wsp-exp8 (deploy: modal deploy wsp/exp8/app.py)
  setup.py                                         one-time: model download/export, benchmark files

The modules import each other as top-level modules (each puts this folder on sys.path), so they work both as
`wsp.exp8.<module>` and inside the Modal containers (/opt/exp8).
"""
