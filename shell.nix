{ pkgs ? import <nixpkgs> {}}:
let
  fhs = pkgs.buildFHSUserEnv {
    name = "example-project-environment";

    targetPkgs = _: [
      pkgs.micromamba
    ];

    profile = ''
      set -e
      export MAMBA_ROOT_PREFIX=${builtins.getEnv "PWD"}/.mamba
      eval "$(micromamba shell hook --shell=bash | sed 's/complete / # complete/g')"

      micromamba create --yes -f micromamba.yml
      micromamba activate my_env1

      set +e
    '';


  };
in fhs.env
