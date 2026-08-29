import { auth } from "@/auth";

export default auth((request) => {
  const isAuthRoute = request.nextUrl.pathname.startsWith("/api/auth");
  if (isAuthRoute || request.auth) {
    return;
  }
  return Response.redirect(new URL("/api/auth/signin", request.nextUrl.origin));
});

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"]
};
