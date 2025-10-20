"""Reusable Tkinter widgets used across the quoting UI."""

from __future__ import annotations

from appkit.ui.tk_compat import tk, ttk


class CreateToolTip:
    """Attach a lightweight tooltip to a Tk widget."""

    def __init__(
        self,
        widget: tk.Widget,
        text: str,
        *,
        delay: int = 500,
        wraplength: int = 320,
    ) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self._after_id: str | None = None
        self._tip_window: tk.Toplevel | None = None
        self._label: tk.Label | ttk.Label | None = None
        self._pinned = False

        self.widget.bind("<Enter>", self._schedule_show, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")
        self.widget.bind("<FocusIn>", self._schedule_show, add="+")
        self.widget.bind("<FocusOut>", self._hide, add="+")
        self.widget.bind("<ButtonPress>", self._on_button_press, add="+")
        self.widget.bind("<Button-1>", self._toggle_pin, add="+")

    def update_text(self, text: str) -> None:
        self.text = text
        if self._label is not None:
            self._label.configure(text=text)

    def _on_button_press(self, event: tk.Event | None = None) -> None:
        if event is not None and getattr(event, "num", None) == 1:
            return
        self._hide()

    def _toggle_pin(self, _event: tk.Event | None = None) -> None:
        if self._pinned:
            self._pinned = False
            self._hide()
        else:
            self._pinned = True
            self._cancel_scheduled()
            self._show()

    def _schedule_show(self, _event: tk.Event | None = None) -> None:
        self._cancel_scheduled()
        if not self.text:
            return
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel_scheduled(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            finally:
                self._after_id = None

    def _show(self) -> None:
        if self._tip_window is not None or not self.text:
            return

        bbox: tuple[int, int, int, int] | None = None
        bbox_method = getattr(self.widget, "bbox", None)
        if callable(bbox_method):
            try:
                raw_bbox = bbox_method("insert")  # type: ignore[arg-type]
                if isinstance(raw_bbox, (tuple, list)) and len(raw_bbox) >= 4:
                    bbox = (
                        int(raw_bbox[0]),
                        int(raw_bbox[1]),
                        int(raw_bbox[2]),
                        int(raw_bbox[3]),
                    )
            except Exception:
                bbox = None
        if bbox:
            x, y, width, height = bbox
        else:
            x = y = 0
            width = self.widget.winfo_width()
            height = self.widget.winfo_height()

        root_x = self.widget.winfo_rootx()
        root_y = self.widget.winfo_rooty()
        x = root_x + x + width + 12
        y = root_y + y + height + 12

        master = self.widget.winfo_toplevel()
        tip = tk.Toplevel(master)
        tip.wm_overrideredirect(True)
        try:
            tip.transient(master)
        except Exception:
            tip.wm_transient(master)
        tip.wm_geometry(f"+{x}+{y}")
        tip.lift()
        try:
            tip.attributes("-topmost", True)
        except Exception:
            pass

        label = tk.Label(
            tip,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("tahoma", 8, "normal"),
            wraplength=self.wraplength,
        )
        label.pack(ipadx=4, ipady=2)

        self._tip_window = tip
        self._label = label

    def _hide(self, _event: tk.Event | None = None) -> None:
        self._cancel_scheduled()
        if self._pinned:
            return
        if self._tip_window is not None:
            self._tip_window.destroy()
            self._tip_window = None
        self._label = None


class ScrollableFrame(ttk.Frame):
    """A ttk frame with vertical scrolling support."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        # Bind wheel only while cursor is over this widget
        self.inner.bind("<Enter>", self._bind_mousewheel)
        self.inner.bind("<Leave>", self._unbind_mousewheel)

    # Windows & macOS wheel
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-int(event.delta / 120), "units")

    # Linux wheel
    def _on_mousewheel_linux(self, event):
        self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

    def _bind_mousewheel(self, _):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")


__all__ = ["CreateToolTip", "ScrollableFrame"]
