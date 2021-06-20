using AgnosticBayesEnsemble
using Documenter

DocMeta.setdocmeta!(AgnosticBayesEnsemble, :DocTestSetup, :(using AgnosticBayesEnsemble); recursive=true)

makedocs(;
    modules=[AgnosticBayesEnsemble],
    authors="Jules Rasetaharison <Jules.rasetaharison@auticon.de>",
    repo="https://github.com/hondoRandale/AgnosticBayesEnsemble.jl/hondoRandale/AgnosticBayesEnsemble.jl/blob/{commit}{path}#{line}",
    sitename="AgnosticBayesEnsemble.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://hondoRandale.github.io/AgnosticBayesEnsemble.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/hondoRandale/AgnosticBayesEnsemble.jl/hondoRandale/AgnosticBayesEnsemble.jl",
)
