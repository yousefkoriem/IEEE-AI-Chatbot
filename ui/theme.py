"""IEEE × CS × CIS blended Gradio theme."""

import gradio as gr


class IEEETheme(gr.themes.Soft):
    """Custom theme blending IEEE, Computer Society, and CIS brand colours.

    Palette:
        Primary   #00629B  IEEE Blue (master brand)
        Secondary #00B5E2  IEEE Cyan (CIS accent)
        Accent    #FFD100  IEEE Yellow (CIS tertiary)
        Bg        #F7F9FC  Soft blue-white
        Text      #1A1A2E  Near-black
    """

    def __init__(self):
        super().__init__(
            primary_hue=gr.themes.Color(
                c50="#E6F2F8",
                c100="#CCE5F1",
                c200="#99CBE3",
                c300="#66B1D5",
                c400="#3397C7",
                c500="#00629B",
                c600="#00527F",
                c700="#004163",
                c800="#003147",
                c900="#00202B",
                c950="#001018",
            ),
            secondary_hue=gr.themes.Color(
                c50="#E6F9FD",
                c100="#CCF3FB",
                c200="#99E7F7",
                c300="#66DBF3",
                c400="#33CFEF",
                c500="#00B5E2",
                c600="#0098BD",
                c700="#007A98",
                c800="#005D73",
                c900="#003F4E",
                c950="#002029",
            ),
            neutral_hue=gr.themes.Color(
                c50="#F7F9FC",
                c100="#EEF2F7",
                c200="#DDE5EF",
                c300="#CCD8E7",
                c400="#AABDD3",
                c500="#88A2BF",
                c600="#6687AB",
                c700="#506D8C",
                c800="#3A536D",
                c900="#24394E",
                c950="#1A1A2E",
            ),
            font=("Montserrat", "ui-sans-serif", "system-ui", "sans-serif"),
            font_mono=("JetBrains Mono", "ui-monospace", "monospace"),
        )

        # Global style overrides
        self.set(
            body_background_fill="#F7F9FC",
            body_text_color="#1A1A2E",
            block_title_text_color="#00629B",
            button_primary_background_fill="#00629B",
            button_primary_background_fill_hover="#004163",
            button_primary_text_color="#FFFFFF",
            button_secondary_background_fill="#00B5E2",
            button_secondary_background_fill_hover="#0098BD",
            button_secondary_text_color="#FFFFFF",
            input_border_color="#CCE5F1",
            input_border_color_focus="#00629B",
            checkbox_background_color="#00629B",
            checkbox_label_text_color="#1A1A2E",
        )


# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------

CUSTOM_CSS = """
/* Google Font import */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&family=JetBrains+Mono&display=swap');

/* Header gradient: IEEE Blue → CIS Cyan */
.gradio-container > .main > .wrap > .contain > .tabs > .tab-nav {
    background: linear-gradient(135deg, #00629B 0%, #00B5E2 100%) !important;
}

/* Chat bubble styles */
.message.bot .message-content {
    border-left: 3px solid #00B5E2 !important;
    background: #F7F9FC !important;
}

.message.user .message-content {
    border-left: 3px solid #00629B !important;
}

/* Confidence badges */
.confidence-high {
    background: #00629B;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

.confidence-medium {
    background: #FFD100;
    color: #1A1A2E;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

.confidence-low {
    background: #E74C3C;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Suggestion chips */
.suggestion-chip {
    display: inline-block;
    background: #E6F9FD;
    border: 1px solid #00B5E2;
    color: #00629B;
    padding: 6px 14px;
    border-radius: 20px;
    margin: 4px;
    cursor: pointer;
    font-size: 0.85rem;
    transition: all 0.2s;
}

.suggestion-chip:hover {
    background: #00B5E2;
    color: white;
}

/* IEEE Yellow accent for highlights */
.highlight-accent {
    color: #FFD100;
}

/* Sidebar status panel */
.status-panel {
    background: linear-gradient(180deg, #00629B 0%, #004163 100%);
    color: white;
    padding: 16px;
    border-radius: 8px;
    font-size: 0.85rem;
}

.status-panel .status-item {
    margin: 4px 0;
}

.status-dot-green { color: #2ECC71; }
.status-dot-red { color: #E74C3C; }
.status-dot-yellow { color: #FFD100; }
"""
