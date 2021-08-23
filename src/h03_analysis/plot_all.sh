
for task in 'pos_tag' 'dep_label'
do
    for lang in 'english' 'marathi' 'turkish' 'basque'
    do
        python src/h03_analysis/plot_pareto.py --task ${task} --language ${lang}
    done
done
