import { auth } from "@/auth";
import { RunConsole } from "@/components/run-console";

export default async function HomePage() {
  const session = await auth();
  const operator = session?.user?.email ?? session?.user?.name ?? "operator";

  return <RunConsole operator={operator} />;
}
