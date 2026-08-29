import NextAuth from "next-auth";

const issuer = process.env.AUTH_OIDC_ISSUER;

export const { auth, handlers } = NextAuth({
  trustHost: true,
  providers: issuer
    ? [
        {
          id: "oidc",
          name: "OIDC",
          type: "oidc",
          issuer,
          clientId: process.env.AUTH_OIDC_CLIENT_ID,
          clientSecret: process.env.AUTH_OIDC_CLIENT_SECRET
        }
      ]
    : []
});
