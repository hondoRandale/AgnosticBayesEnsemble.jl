using Documenter, AgnosticBayesEnsemble

makedocs( modules=[AgnosticBayesEnsemble],
          doctest=true, 
          sitename="AgnosticBayesEnsemble", 
          format = :html )

deploydocs( deps = Deps.pip("mkdocs", "python-markdown-math"),
            repo = "github.com/hondoRandale/AgnosticBayesEnsemble.jl",
            julia  = "1.2.0",
            osname = "linux" )        