import lime
import lime.lime_tabular


# define a function for lime - local interpretable model-agnostic explanations
def lime_explanation(model, X_train, X_test, class_names, chosen_index):
    import lime
    import lime.lime_tabular

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=class_names,
        kernel_width=5,
    )
    choosen_instance = X_test.loc[[chosen_index]].values[0]
    exp = explainer.explain_instance(
        choosen_instance,
        lambda x: model.predict_proba(x).astype(float),
        num_features=10,
    )
    fig = exp.as_pyplot_figure()
    return fig
