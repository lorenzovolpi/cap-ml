import dash_bootstrap_components as dbc

APP_NAME = "QuAcc"


def get_navbar():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink(["plot"], href="/plot", target="_blank")),
            dbc.NavItem(dbc.NavLink(["table"], href="/table", target="_blank")),
        ],
        brand=APP_NAME,
        brand_href="/",
        color="dark",
        dark=True,
    )

    return navbar
