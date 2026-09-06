"""Optional AppKit toolbar; the subtitle editor remains in its WKWebView.

All Cocoa work runs on the AppKit main thread. Native actions dispatch a fixed
command vocabulary; document paths and executable JavaScript never cross back
from toolbar state updates.
"""

from __future__ import annotations

import json
import logging
import sys
import threading

log = logging.getLogger(__name__)
_controller = None


def normalize_state(value: object) -> dict:
    """Keep untrusted bridge input small, typed and presentation-only."""
    data = value if isinstance(value, dict) else {}
    return {
        "title": str(data.get("title") or "SubForge")[:180],
        "status": str(data.get("status") or "字幕工作室")[:90],
        "appearance": data.get("appearance")
        if data.get("appearance") in ("light", "dark")
        else "system",
        "can_export": data.get("can_export") is True,
        "can_inspect": data.get("can_inspect") is True,
        "inspector_open": data.get("inspector_open") is True,
        "running": data.get("running") is True,
    }


def get_desktop_state() -> dict:
    controller = _controller
    return dict(controller.capabilities) if controller is not None else {"toolbar": False}


def sync_desktop_state(value: object) -> dict:
    controller = _controller
    if controller is None:
        return {"toolbar": False}
    from PyObjCTools import AppHelper

    AppHelper.callAfter(controller.applyState, normalize_state(value))
    return get_desktop_state()


def install_native_toolbar(window) -> bool:
    """Install after the webview is ready, retaining the original window owner."""
    if _controller is not None:
        return bool(get_desktop_state().get("toolbar"))
    if sys.platform != "darwin":
        return False
    import AppKit
    import objc
    from PyObjCTools import AppHelper

    ready = threading.Event()

    class SubForgeToolbarController(
        AppKit.NSObject, protocols=[objc.protocolNamed("NSToolbarDelegate")]
    ):
        @objc.python_method
        def setup(self):
            self.window = window.native
            self.webview = self.window.contentView()
            self.buttons = {}
            self.items = {}
            self.capabilities = {"toolbar": False, "liquid_glass": False}
            self.readAccessibility()
            toolbar = AppKit.NSToolbar.alloc().initWithIdentifier_("SubForge.Workspace")
            toolbar.setDelegate_(self)
            toolbar.setAllowsUserCustomization_(False)
            toolbar.setAutosavesConfiguration_(False)
            toolbar.setDisplayMode_(AppKit.NSToolbarDisplayModeIconOnly)
            self.toolbar = toolbar
            self.window.setToolbarStyle_(AppKit.NSWindowToolbarStyleUnified)
            self.window.setTitlebarAppearsTransparent_(True)
            self.window.setTitlebarSeparatorStyle_(AppKit.NSTitlebarSeparatorStyleNone)
            self.window.setToolbar_(toolbar)
            self.capabilities["toolbar"] = True
            self.applyState(normalize_state({}))
            AppKit.NSWorkspace.sharedWorkspace().notificationCenter().addObserver_selector_name_object_(
                self,
                "accessibilityChanged:",
                AppKit.NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification,
                None,
            )
            log.info("Native toolbar installed: %s", self.capabilities)
            log.info(
                "Native button metrics: %s",
                {
                    key: {
                        "bezel": button.bezelStyle(),
                        "bordered": bool(button.isBordered()),
                        "size": str(button.frame().size),
                    }
                    for key, button in self.buttons.items()
                },
            )

        @objc.python_method
        def readAccessibility(self):
            workspace = AppKit.NSWorkspace.sharedWorkspace()
            self.capabilities.update(
                {
                    "reduce_transparency": bool(
                        workspace.accessibilityDisplayShouldReduceTransparency()
                    ),
                    "reduce_motion": bool(workspace.accessibilityDisplayShouldReduceMotion()),
                    "increase_contrast": bool(
                        workspace.accessibilityDisplayShouldIncreaseContrast()
                    ),
                }
            )

        def accessibilityChanged_(self, notification):
            self.readAccessibility()
            self.sendEvent("subforge:desktop", self.capabilities)

        @objc.python_method
        def sendEvent(self, event, detail):
            # Use WKWebView directly here. pywebview.evaluate_js waits for a main-
            # thread callback, which would deadlock an AppKit button action.
            script = f"window.dispatchEvent(new CustomEvent({json.dumps(event)},{{detail:{json.dumps(detail)}}}));"
            self.webview.evaluateJavaScript_completionHandler_(script, None)

        def command_(self, sender):
            command = {1: "sidebar", 2: "import", 3: "inspector", 4: "export", 5: "cancel"}.get(
                sender.tag()
            )
            if command:
                self.window.makeFirstResponder_(self.webview)
                self.sendEvent("subforge:command", command)

        @objc.python_method
        def button(self, command, label, symbol):
            image = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                symbol, label
            )
            button = AppKit.NSButton.buttonWithTitle_image_target_action_(
                label, image, self, "command:"
            )
            button.setTag_(command)
            button.setControlSize_(AppKit.NSControlSizeLarge)
            button.setFont_(AppKit.NSFont.systemFontOfSize_weight_(13, AppKit.NSFontWeightMedium))
            button.setImageHugsTitle_(True)
            button.setSymbolConfiguration_(
                AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(
                    14, AppKit.NSFontWeightRegular
                )
            )
            button.setButtonType_(
                AppKit.NSButtonTypePushOnPushOff
                if command == 3
                else AppKit.NSButtonTypeMomentaryPushIn
            )
            button.setBordered_(True)
            modern = button.respondsToSelector_("setTintProminence:")
            # Use the native glass button bezel: it owns padding, hit feedback,
            # focus and disabled treatments. No nested glass backplate or frames.
            button.setBezelStyle_(
                AppKit.NSBezelStyleGlass if modern else AppKit.NSBezelStyleTexturedRounded
            )
            if modern:
                button.setPrefersCompactControlSizeMetrics_(False)
                button.setTintProminence_(AppKit.NSTintProminenceAutomatic)
                self.capabilities["liquid_glass"] = True
            button.setToolTip_(
                {
                    1: "显示或隐藏侧边栏",
                    2: "导入视频、音频或字幕（⌘O）",
                    3: "显示或隐藏处理选项",
                    4: "选择格式并导出字幕",
                    5: "取消当前任务",
                }[command]
            )
            button.setAccessibilityLabel_(label or button.toolTip())
            button.sizeToFit()
            # Keep native intrinsic metrics and enough optical padding for
            # localized titles, including builds running on older macOS.
            text_size = AppKit.NSString.stringWithString_(label).sizeWithAttributes_(
                {AppKit.NSFontAttributeName: button.font()}
            )
            width = max(button.frame().size.width, text_size.width + (56 if label else 40))
            height = max(button.frame().size.height, 36)
            button.setFrameSize_(AppKit.NSMakeSize(width, height))
            self.buttons[command] = button
            return button

        def toolbarDefaultItemIdentifiers_(self, toolbar):
            return [
                "sidebar",
                AppKit.NSToolbarFlexibleSpaceItemIdentifier,
                "import",
                "processing",
                AppKit.NSToolbarSpaceItemIdentifier,
                "cancel",
                "export",
            ]

        def toolbarAllowedItemIdentifiers_(self, toolbar):
            return self.toolbarDefaultItemIdentifiers_(toolbar)

        def toolbar_itemForItemIdentifier_willBeInsertedIntoToolbar_(
            self, toolbar, identifier, inserted
        ):
            specs = {
                "sidebar": (1, "", "sidebar.left"),
                "import": (2, "导入", "square.and.arrow.down"),
                "processing": (3, "处理选项", "slider.horizontal.3"),
                "cancel": (5, "取消任务", "stop.circle"),
                "export": (4, "导出", "square.and.arrow.up"),
            }
            if identifier not in specs:
                return None
            command, label, symbol = specs[identifier]
            item = AppKit.NSToolbarItem.alloc().initWithItemIdentifier_(identifier)
            item.setLabel_(label or "侧边栏")
            item.setBordered_(False)
            button = self.button(command, label, symbol)
            # A layout-only host keeps the explicit native button bezel and
            # intrinsic metrics intact, without adding a second glass surface.
            host = AppKit.NSView.alloc().initWithFrame_(button.frame())
            host.addSubview_(button)
            button.setTranslatesAutoresizingMaskIntoConstraints_(False)
            AppKit.NSLayoutConstraint.activateConstraints_(
                [
                    button.leadingAnchor().constraintEqualToAnchor_(host.leadingAnchor()),
                    button.trailingAnchor().constraintEqualToAnchor_(host.trailingAnchor()),
                    button.topAnchor().constraintEqualToAnchor_(host.topAnchor()),
                    button.bottomAnchor().constraintEqualToAnchor_(host.bottomAnchor()),
                    button.widthAnchor().constraintGreaterThanOrEqualToConstant_(
                        button.frame().size.width
                    ),
                    button.heightAnchor().constraintGreaterThanOrEqualToConstant_(
                        button.frame().size.height
                    ),
                ]
            )
            item.setView_(host)
            item.setTag_(command)
            item.setTarget_(self)
            item.setAction_("command:")
            item.setAutovalidates_(False)
            if identifier == "sidebar":
                item.setNavigational_(True)
            self.items[identifier] = item
            return item

        @objc.python_method
        def setCancelVisible(self, visible):
            item = self.items.get("cancel")
            if item is not None and item.respondsToSelector_("setHidden:"):
                item.setHidden_(not visible)
                return
            identifiers = [item.itemIdentifier() for item in self.toolbar.items()]
            if visible and "cancel" not in identifiers:
                self.toolbar.insertItemWithItemIdentifier_atIndex_(
                    "cancel", identifiers.index("export")
                )
            elif not visible and "cancel" in identifiers:
                self.toolbar.removeItemAtIndex_(identifiers.index("cancel"))

        @objc.python_method
        def applyState(self, data):
            self.window.setTitle_(data["title"])
            self.window.setSubtitle_(data["status"])
            appearance = {
                "light": AppKit.NSAppearanceNameAqua,
                "dark": AppKit.NSAppearanceNameDarkAqua,
            }.get(data["appearance"])
            self.window.setAppearance_(
                AppKit.NSAppearance.appearanceNamed_(appearance) if appearance else None
            )
            for key, enabled in (
                (4, data["can_export"]),
                (3, data["can_inspect"]),
                (5, data["running"]),
            ):
                if key in self.buttons:
                    self.buttons[key].setEnabled_(enabled)
                    identifier = {3: "processing", 4: "export", 5: "cancel"}[key]
                    self.items[identifier].setEnabled_(enabled)
            self.setCancelVisible(data["running"])
            if 3 in self.buttons:
                inspector = self.buttons[3]
                selected = data["inspector_open"] and data["can_inspect"]
                inspector.setState_(
                    AppKit.NSControlStateValueOn if selected else AppKit.NSControlStateValueOff
                )
                inspector.setToolTip_("收起处理选项" if selected else "展开处理选项")
                if inspector.respondsToSelector_("setTintProminence:"):
                    inspector.setTintProminence_(AppKit.NSTintProminenceNone)
                    inspector.setBezelColor_(AppKit.NSColor.controlColor() if selected else None)
            if 4 in self.buttons:
                export = self.buttons[4]
                if export.respondsToSelector_("setTintProminence:"):
                    export.setBezelColor_(None)
                    export.setTintProminence_(
                        AppKit.NSTintProminencePrimary
                        if data["can_export"]
                        else AppKit.NSTintProminenceAutomatic
                    )
                    foreground = (
                        AppKit.NSColor.whiteColor()
                        if data["can_export"]
                        else AppKit.NSColor.labelColor()
                    )
                    export.setAttributedTitle_(
                        AppKit.NSAttributedString.alloc().initWithString_attributes_(
                            "导出",
                            {
                                AppKit.NSFontAttributeName: export.font(),
                                AppKit.NSForegroundColorAttributeName: foreground,
                            },
                        )
                    )
                    metrics = AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(
                        14, AppKit.NSFontWeightRegular
                    )
                    colors = AppKit.NSImageSymbolConfiguration.configurationWithPaletteColors_(
                        [foreground]
                    )
                    export.setSymbolConfiguration_(
                        metrics.configurationByApplyingConfiguration_(colors)
                    )
                else:
                    export.setBezelColor_(
                        AppKit.NSColor.controlAccentColor() if data["can_export"] else None
                    )

    def install():
        global _controller
        controller = SubForgeToolbarController.alloc().init()
        try:
            controller.setup()
            _controller = controller
        except Exception:
            log.exception("Native toolbar unavailable; retaining the web toolbar")
            if getattr(window, "native", None):
                window.native.setToolbar_(None)
        finally:
            ready.set()

    AppHelper.callAfter(install)
    ready.wait(5)
    return bool(get_desktop_state().get("toolbar"))
