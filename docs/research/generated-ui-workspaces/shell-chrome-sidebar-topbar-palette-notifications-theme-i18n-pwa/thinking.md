# Raw Reasoning

This final workspace is cross-cutting. It should not remodel every page; it should make the persistent shell behave like a mature generated application frame. The existing WP2/WP6/WP7 foundation already provided sidebar, topbar, theme, locale, PWA, and command palette pieces. The issue was that several pieces were either hidden, transient, or only partially accessible.

The command palette already searched generated records, but only users who knew Ctrl/Cmd-K could find it. A visible topbar trigger solves discoverability without changing search behavior. Adding dialog semantics improves assistive technology expectations.

Transient toasts are useful but insufficient for a serious app shell because users can miss them. A lightweight notification history panel gives a stable place for toast, offline/online, install, and update events without adding backend notification infrastructure.

PWA support existed but was invisible. Install and update buttons remain hidden until browser/service-worker events make them relevant. This preserves visual density while making the capability discoverable.

The sidebar active state should come from the server-rendered request path whenever Flask is serving a real page. This avoids relying only on client-side topnav scripts and gives screen readers `aria-current` immediately.

Rejected ideas:

- Push notifications: requires permissions, external browser behavior, and backend event semantics outside generated apps.
- Replacing the command palette search implementation: existing `/api/search` is adequate; the high-value gap was access and semantics.
- Full i18n language names: useful polish, but this pass prioritized shell function and PWA completion without expanding the translation catalog.
