.PHONY: install audit smoke pipeline verify test
install:
	python -m pip install -e '.[full,test]'
audit:
	python scripts/01_data_audit.py --out outputs
smoke:
	python scripts/11_run_all_pipeline.py --fast-debug --confirm-final-test --out outputs/smoke
pipeline:
	python scripts/11_run_all_pipeline.py --confirm-final-test --out outputs
verify:
	python scripts/12_release_audit.py --compare outputs
test:
	python -m pytest -q
