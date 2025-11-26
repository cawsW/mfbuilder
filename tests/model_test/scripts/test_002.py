from mfbuilder.mfmain import LoaderYaml, Director

def main():
    config = "./configs/test_002.yaml"
    cfg = LoaderYaml(config).read_yml()
    director = Director()
    director.build(cfg)
    print(f"Built {cfg.base.engine} model '{cfg.base.name}' in {cfg.base.workspace_path}")

if __name__ == "__main__":
    main()
