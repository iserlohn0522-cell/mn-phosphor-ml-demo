from pathlib import Path

from phosphor_ml.config import Config


def test_config_loads_placeholder_env_without_real_api_keys(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENALEX_EMAIL=placeholder@example.com",
                "OPENALEX_API_KEY=oa_demo_key",
                "SEMANTIC_SCHOLAR_API_KEY=",
                "CROSSREF_MAILTO=",
                "DATA_DIR=data",
                "LOG_LEVEL=DEBUG",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = Config.from_env(env_file=env_file, project_root=tmp_path)

    assert config.openalex_email == "placeholder@example.com"
    assert config.openalex_api_key == "oa_demo_key"
    assert config.semantic_scholar_api_key == ""
    assert config.crossref_mailto == ""
    assert config.data_dir == Path(tmp_path / "data")
    assert config.log_level == "DEBUG"
