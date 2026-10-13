"""Streamlit UI for OLAP operations. Used by pages/8_OLAP_Operations.py and as a tab in pages/6_Visualization.py."""

import plotly.express as px
import streamlit as st

from utils import olap


def _sanitize_multi(key, options):
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in options]


def _sanitize_single(key, options):
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def add_table_to_report(title, table, kind):
    """Store an OLAP result snapshot so it appears in the PDF report."""
    charts = st.session_state.setdefault("report_charts", [])
    spec = {"type": "OLAP table", "title": title, "table": table.copy(), "kind": kind}
    if any(c["type"] == "OLAP table" and c["title"] == title for c in charts):
        st.info("This OLAP result is already in the PDF report.")
    else:
        charts.append(spec)
        st.success("Added to the PDF report (download it from the Results page).")


def _bar(result, group_columns, value_col, title):
    data = result.head(60)
    if len(group_columns) == 1:
        fig = px.bar(data, x=group_columns[0], y=value_col, hover_data=["Rows"], title=title, template="plotly_white")
    elif len(group_columns) == 2:
        fig = px.bar(data, x=group_columns[0], y=value_col, color=group_columns[1], barmode="group",
                     hover_data=["Rows"], title=title, template="plotly_white")
    else:
        data = data.copy()
        data["Path"] = data[group_columns].agg(olap.SEP.join, axis=1)
        fig = px.bar(data, x="Path", y=value_col, hover_data=["Rows"], title=title, template="plotly_white")
    if len(result) > 60:
        st.caption(f"Chart shows the first 60 of {len(result)} groups; the table below has all of them.")
    return fig


def _show_result(result, group_columns, value_col, title, key, kind="bar"):
    """Table + chart + download + add-to-report for a grouped result."""
    if group_columns:
        st.plotly_chart(_bar(result, group_columns, value_col, title), use_container_width=True,
                        key=f"{key}_chart")
    st.dataframe(result, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download result (CSV)", result.to_csv(index=False).encode("utf-8"),
                       file_name=f"{key}.csv", mime="text/csv", key=f"{key}_dl")
    if group_columns and c2.button("➕ Add to PDF report", key=f"{key}_add"):
        add_table_to_report(title, result[group_columns + [value_col]], "bar")


def render_olap(df, key_prefix="olap"):
    k = key_prefix
    st.subheader("OLAP operations")
    st.caption(
        "Your table is treated as an in-memory cube: **dimensions** are categorical or low-cardinality "
        "columns, **measures** are numeric columns. Nothing is written back to your dataset."
    )

    measures = olap.measure_candidates(df)
    if not measures:
        st.warning("OLAP needs at least one numeric column to use as a measure.")
        return

    # ---------------- cube setup ----------------
    with st.expander("⚙️ Cube setup — measure, aggregation, derived dimensions", expanded=True):
        c1, c2 = st.columns(2)
        measure = c1.selectbox("Measure (numeric)", measures, key=f"{k}_measure")
        agg = c2.selectbox("Aggregation", list(olap.AGG_OPTIONS), key=f"{k}_agg")

        _sanitize_multi(f"{k}_bin_cols", measures)
        d1, d2, d3 = st.columns(3)
        bin_cols = d1.multiselect("Bin numeric columns into ranges (new dimensions)", measures,
                                  key=f"{k}_bin_cols")
        n_bins = d2.slider("Number of bins", 2, 10, 4, key=f"{k}_nbins")
        bin_method = d3.selectbox("Binning method", ["Equal width", "Equal frequency"], key=f"{k}_binmethod")

        non_numeric = [c for c in df.columns if c not in measures]
        date_choice = st.selectbox(
            "Expand a date column into Year / Quarter / Month / Day (optional)",
            ["(None)"] + non_numeric, key=f"{k}_date",
        )

    cube, derived, warnings = olap.derive_dimensions(
        df, bin_cols, n_bins, bin_method, None if date_choice == "(None)" else date_choice
    )
    for w in warnings:
        st.warning(w)

    dims = olap.dimension_candidates(cube)
    if not dims:
        st.warning("No suitable dimension columns found (columns with ≤ 100 distinct values).")
        return

    if derived:
        st.caption("Derived dimensions available: " + ", ".join(derived))

    tabs = st.tabs(["🔪 Slice", "🎲 Dice", "⬇️⬆️ Drill-down / Roll-up", "🔄 Pivot"])
    value_col = olap.value_column_name(measure, agg)

    # ---------------- SLICE ----------------
    with tabs[0]:
        st.markdown("**Slice** fixes *one* dimension to *one* value and keeps the rest of the cube.")
        _sanitize_single(f"{k}_slice_dim", dims)
        s1, s2 = st.columns(2)
        dim = s1.selectbox("Dimension to slice", dims, key=f"{k}_slice_dim")
        options = olap.members(cube[dim])
        _sanitize_single(f"{k}_slice_val", options)
        value = s2.selectbox("Keep only this value", options, key=f"{k}_slice_val")
        sliced = olap.slice_cube(cube, dim, value)

        m1, m2 = st.columns(2)
        m1.metric("Rows in slice", f"{len(sliced):,}")
        m2.metric("Rows in cube", f"{len(cube):,}")

        others = [d for d in dims if d != dim]
        if others:
            _sanitize_single(f"{k}_slice_by", others)
            by = st.selectbox("Then group the slice by", others, key=f"{k}_slice_by")
            result = olap.aggregate(sliced, [by], measure, agg)
            _show_result(result, [by], value_col, f"Slice: {dim} = {value} — {value_col} by {by}", f"{k}_slice")
        with st.expander("Rows in the slice"):
            st.dataframe(sliced.head(500), use_container_width=True)
            st.download_button("⬇️ Download slice (CSV)", sliced.to_csv(index=False).encode("utf-8"),
                               file_name="olap_slice.csv", mime="text/csv", key=f"{k}_slice_rows_dl")

    # ---------------- DICE ----------------
    with tabs[1]:
        st.markdown("**Dice** selects a *sub-cube*: several values on *two or more* dimensions at once.")
        st.session_state.setdefault(f"{k}_dice_dims", dims[:2])
        _sanitize_multi(f"{k}_dice_dims", dims)
        chosen = st.multiselect("Dimensions to dice on (2 or more)", dims, key=f"{k}_dice_dims")
        selections = {}
        cols = st.columns(min(max(len(chosen), 1), 3))
        for i, dim in enumerate(chosen):
            options = olap.members(cube[dim])
            st.session_state.setdefault(f"{k}_dice_{dim}", options[: min(3, len(options))])
            _sanitize_multi(f"{k}_dice_{dim}", options)
            selections[dim] = cols[i % len(cols)].multiselect(f"{dim}", options, key=f"{k}_dice_{dim}")

        if len(chosen) < 2:
            st.info("Pick at least two dimensions to dice.")
        elif any(len(v) == 0 for v in selections.values()):
            st.info("Choose at least one value for every selected dimension.")
        else:
            diced = olap.dice_cube(cube, selections)
            m1, m2 = st.columns(2)
            m1.metric("Rows in sub-cube", f"{len(diced):,}")
            m2.metric("Rows in cube", f"{len(cube):,}")
            if diced.empty:
                st.warning("No rows match this combination.")
            else:
                result = olap.aggregate(diced, chosen[:3], measure, agg)
                _show_result(result, chosen[:3], value_col,
                             f"Dice on {', '.join(chosen[:3])} — {value_col}", f"{k}_dice")
                with st.expander("Rows in the sub-cube"):
                    st.dataframe(diced.head(500), use_container_width=True)

    # ---------------- DRILL / ROLL ----------------
    with tabs[2]:
        st.markdown(
            "**Drill-down** moves to finer detail along a hierarchy; **roll-up** aggregates back to a higher "
            "level. Choose the hierarchy from the *top* level to the *most detailed* level — the order you "
            "select the dimensions is the order of the levels."
        )
        _sanitize_multi(f"{k}_hier", dims)
        hierarchy = st.multiselect("Hierarchy (top → bottom)", dims, key=f"{k}_hier",
                                   help="Example: Country, State, City — or the Year / Quarter / Month "
                                        "dimensions created from a date column.")
        if not hierarchy:
            st.info("Select at least one dimension to build a hierarchy.")
        else:
            level_key = f"{k}_level"
            if level_key not in st.session_state:
                st.session_state[level_key] = 1
            st.session_state[level_key] = max(0, min(st.session_state[level_key], len(hierarchy)))

            def _down():
                st.session_state[level_key] = min(st.session_state[level_key] + 1, len(hierarchy))

            def _up():
                st.session_state[level_key] = max(st.session_state[level_key] - 1, 0)

            level = st.session_state[level_key]
            b1, b2, b3 = st.columns([1, 1, 3])
            b1.button("⬆️ Roll-up", on_click=_up, disabled=level == 0, key=f"{k}_up")
            b2.button("⬇️ Drill-down", on_click=_down, disabled=level == len(hierarchy), key=f"{k}_down")
            names = ["All (grand total)"] + hierarchy
            b3.markdown(f"**Current level {level}:** {names[level]}  \n"
                        f"Path: {' → '.join(names[: level + 1])}")

            parent = None
            if level >= 2:
                options = olap.parent_options(cube, hierarchy, level)
                labels = ["(All)"] + [olap.SEP.join(o) for o in options]
                pick = st.selectbox(f"Drill into one {olap.SEP.join(hierarchy[:level - 1])} member",
                                    labels, key=f"{k}_parent_{level}_{'|'.join(hierarchy)}")
                if pick != "(All)":
                    parent = options[labels.index(pick) - 1]

            result = olap.drill(cube, hierarchy, level, measure, agg, parent)
            group_columns = hierarchy[:level]
            title = f"{'Roll-up' if level == 0 else 'Drill-down'} level {level} — {value_col}"
            if parent:
                title += f" within {olap.SEP.join(parent)}"
            _show_result(result, group_columns, value_col, title, f"{k}_drill")

    # ---------------- PIVOT ----------------
    with tabs[3]:
        st.markdown("**Pivot** rotates the cube: pick dimensions for the rows and one for the columns.")
        st.session_state.setdefault(f"{k}_pivot_rows", dims[:1])
        _sanitize_multi(f"{k}_pivot_rows", dims)
        rows = st.multiselect("Row dimension(s) (1–3)", dims, max_selections=3, key=f"{k}_pivot_rows")
        col_options = ["(None)"] + [d for d in dims if d not in rows]
        _sanitize_single(f"{k}_pivot_col", col_options)
        column = st.selectbox("Column dimension", col_options, index=min(1, len(col_options) - 1),
                              key=f"{k}_pivot_col")
        margins = st.checkbox("Show totals", value=True, key=f"{k}_pivot_margins")

        if not rows:
            st.info("Choose at least one row dimension.")
        else:
            try:
                table = olap.pivot(cube, rows, None if column == "(None)" else column, measure, agg, margins)
            except Exception as error:
                st.error(f"Could not build the pivot table: {error}")
                table = None
            if table is not None:
                flat = olap.flatten_index(table)
                st.dataframe(flat, use_container_width=True)
                body = flat.drop(index="Total", errors="ignore").drop(columns="Total", errors="ignore")
                title = f"Pivot — {value_col}: {', '.join(rows)}" + ("" if column == "(None)" else f" × {column}")
                if body.shape[0] and body.shape[1] > 1:
                    fig = px.imshow(body.astype(float), text_auto=".3s", aspect="auto",
                                    color_continuous_scale="Blues", title=title, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True, key=f"{k}_pivot_chart")
                c1, c2 = st.columns(2)
                c1.download_button("⬇️ Download pivot (CSV)", flat.to_csv().encode("utf-8"),
                                   file_name="olap_pivot.csv", mime="text/csv", key=f"{k}_pivot_dl")
                if c2.button("➕ Add to PDF report", key=f"{k}_pivot_add"):
                    add_table_to_report(title, body, "heatmap" if body.shape[1] > 1 else "bar")
