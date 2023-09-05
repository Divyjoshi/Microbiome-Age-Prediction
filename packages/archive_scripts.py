def generate_comparison_chart(df, level, column_cat, taxa, age_group, x_cmap, x_order, path, p_value_tresh=0.05):

    save_dir = os.path.join(path, f"L{level}", age_group)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # if not os.path.isdir(path):
    #     os.mkdir(path)
    p_value=KS_test(df,list(df[column_cat].unique()),column_cat,taxa)

    if p_value != "not performed" and float(p_value)<p_value_tresh:
        print(p_value)

        fig = px.box(
            df, x=column_cat, y=taxa,
            labels={taxa: taxa},
            color=column_cat,
            color_discrete_map=x_cmap,
            category_orders=x_order,
            points="all",
            template="plotly_white+presentation"
        )

        add_template_and_save(
            fig,
            title=f"{taxa} Abundance Distribution for {column_cat}",
            chart_file_name=os.path.join(
                save_dir, f"{taxa}.html"
            ))
        
        return taxa