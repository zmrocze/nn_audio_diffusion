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
      # micromamba create --yes -q -n my-mamba-environment
      # micromamba activate my-mamba-environment
      # # micromamba install --yes -f micromamba_conf.txt -c conda-forge
      # micromamba install --yes -f micromamba_conf_2.txt -c conda-forge
      # micromamba install --yes -f micromamba_conf_3.txt -c conda-forge
      # micromamba install --yes -f micromamba_conf_4.txt -c conda-forge
      
      micromamba create --yes -f micromamba.yml
      micromamba activate my_env1

      set +e
    '';


  };
in fhs.env
